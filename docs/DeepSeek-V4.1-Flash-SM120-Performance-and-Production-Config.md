# DeepSeek-V4.1-Flash on RTX PRO 6000 (SM120) — Measured Performance & Production Configuration

> Status: **In production** as the local inference backend for a single-user Hermes deployment (2026-09-23), serving a 1M-context assistant workload.
> Supersedes the performance claims in [DeepSeek-V4.1-Flash-SM120-Status.md](DeepSeek-V4.1-Flash-SM120-Status.md) (2026-09-22) — see [Corrections to prior notes](#corrections-to-prior-notes).
> Related: [GLM-5.3-Flash deployment guide](GLM-5.3-Flash-SM120-Deployment.md).

## 1. Hardware

| Item | Value |
|---|---|
| GPUs | 8× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB each |
| Interconnect | PCIe, **no NVLink** (this bounds every throughput number below) |
| Compute capability | sm_120 / CC 12.0, driver 590.48.01 |
| Storage | weights on `/ssd` (~64 GB/card at TP8) |
| Serving engine | SGLang (container `deepseek-v41-4x6000:local`, host network, port 8000) |
| Client path | SSH tunnel to Mac, exposed as OpenAI-compatible endpoint |

## 2. Final production configuration

Container env (authoritative — the launcher reads these and passes them through):

```bash
docker run --gpus all -d --name dsv41 \
  --shm-size=64g --ipc=host --network host \
  --restart unless-stopped \
  -e TP_SIZE=8 -e DP_SIZE=1 -e EP_SIZE=8 \
  -e SPEC_ALGORITHM=dspark -e DSPARK_BLOCK_SIZE=6 \
  -e CHUNKED_PREFILL_SIZE=2048 \
  -v /ssd/DeepSeek-V4.1-Flash:/model \
  deepseek-v41-4x6000:local
```

Effective `server_args` verified at runtime via `/get_server_info`:

| Parameter | Value |
|---|---|
| `tp_size` / `ep_size` / `dp_size` | **8 / 8 / 1** |
| `speculative_algorithm` | **DSPARK**, `speculative_dspark_block_size = 6` |
| `chunked_prefill_size` | 2048 |
| `context_length` | 1,048,576 |
| `kv_cache_dtype` | `fp8_e4m3` |
| `mem_fraction_static` | 0.72 |
| `max_running_requests` | 16 |
| `attention_backend` | `dsv4` |
| `moe_runner_backend` | `flashinfer_mxfp4` |
| CUDA graph — decode | `full`, `max_bs = 16` |
| CUDA graph — prefill | **disabled** (see §6) |

**Topology is not a free parameter.** TP8/EP8 is the only layout that starts on this model — see §6.

## 3. Methodology (so numbers are comparable)

- **Prefill**: unique random prompt (no radix/prefix-cache reuse), `max_tokens=1`, cold measurement; tokens counted from the server's `usage.prompt_tokens`, not from SSE chunk counts.
- **Decode**: streaming request with `stream_options.include_usage=true`; token counts from server usage; `decode_rate = completion_tokens / (total − TTFT)`; 3 reps, median reported.
- **Warm-path test**: identical prompt re-sent to trigger the radix cache.
- **Config verification before any number counts**: after each restart, `/get_server_info` must report the intended `speculative_algorithm`/`block_size` **and** `/v1/models` must return 200. A leg that fails this check is discarded rather than recorded (see [§7](#7-operational-lessons)).
- Everything below was measured through the real client path (SSH tunnel), not on a co-located loopback.

## 4. Measured results

### 4.1 Cold single-stream prefill — 6.8–7.2K tok/s, PCIe-bound

| Configuration | 128K cold prefill (tok/s) |
|---|---|
| **`chunked_prefill_size=2048` (production)** | **6,853** |
| production re-run (repeatability check) | 6,765 (−1.3%) |
| flashmla prefill backend | 6,845 |
| fused-MoE all-reduce | 6,828 |
| `chunked_prefill_size=4096` | 6,301 |
| `chunked_prefill_size=1024` | 6,012 |

- 32K prompt: ≈ 7,196 tok/s (prefill rate rises slightly as fixed overhead amortises).
- **Conclusion: ~7K tok/s is the practical ceiling** for this model/hardware. GPU utilization reached ~87% at 220 W of a 600 W budget — the bottleneck is cross-GPU communication (PCIe, no NVLink inside a TP8 group), not compute.

### 4.2 Prefix-cache reuse is the only large multiplier

| Metric | Value |
|---|---|
| Cache-hit prefill latency | **0.34 s** (≈ **325K tok/s** effective, **≈39×** cold rate) |
| 5.2K prompt, cold → warm (live client path) | 777 ms → 208 ms (**3.7×**; short prefix, so partial reuse) |

For an agent workload that re-sends a long, mostly-identical prefix every turn, this — not raw prefill — is what dominates perceived latency. Keep the prefix stable; never mutate system prompt or history mid-conversation.

### 4.3 DSPARK speculative decoding: A/B with gamma=2

Clean A/B on the same container/launcher, each leg verified by `server_info` (spec/block) and by engine-log acceptance length:

| Prompt class | DSPARK ON (`block_size=2`) | DSPARK OFF | Gain |
|---|---|---|---|
| prose (natural language), temp 0 | 76.5 tok/s | 36.4 tok/s | **2.1×** |
| prose, temp 0.7 | 78.9 tok/s | 36.5 tok/s | 2.2× |
| structured (JSON-like), temp 0 | 120.3 tok/s | 36.5 tok/s | **3.3×** |
| structured, temp 0.7 | 120.5 tok/s | 36.5 tok/s | 3.3× |
| acceptance length (`accept len`) | 3.0 tokens/step | 1.0 (= no speculation) | — |

With DSPARK disabled the model flattens to **~36.5 tok/s regardless of prompt class** — that is the non-speculative decode ceiling for this 769 B-class MoE on PCIe. Speculative decoding is therefore not optional: it is the difference between usable and unusable interactive latency.

### 4.4 Gamma (DSPARK block size) sweep

| Block size | prose | structured | Note |
|---|---|---|---|
| **6 (production)** | 79.4–80.1 | **236.8–252.3** | measured in-place, no restart → clean |
| 2 | 76.5–78.9 | 120.3–120.5 | clean re-test (§4.3) |
| 4 / 8 / 12 | — | — | **not usable**: earliest runs used a launcher without per-leg isolation; a stale container can hold port 8000 and silently serve an *older* config, so those numbers are mislabelled and must be re-measured before citation |

**Decision: keep `DSPARK_BLOCK_SIZE=6`.** Block size 2 halves structured-output throughput relative to production; no evidence supports changing the shipped value.

### 4.5 Live end-to-end check (through the SSH tunnel, production config)

| Metric | Result |
|---|---|
| Decode, prose | 93.9 tok/s (91.8 / 93.9 / 96.6), temp 0 |
| Decode, structured | 134.1 tok/s (126.4 / 134.1 / 137.0) |
| Decode, prose @ temp 0.7 | 86.4 tok/s |
| TTFT | **143–169 ms** |
| Cold prefill (2.7K / 5.2K / 21K tokens) | 5,828 / 6,736 / 7,017 tok/s |

## 5. Comparison with GLM-5.3-Flash (historical, same hardware)

The comparison is built from that deployment's vLLM engine logs (`Avg prompt/generation throughput` lines, 234 single-request windows), because the two models cannot be co-resident on 8 GPUs.

| Metric | DSV4.1-Flash (SGLang, measured) | GLM-5.3-Flash (vLLM, historical logs) |
|---|---|---|
| Cold prefill, single stream | **6,853 tok/s** (controlled) | **no cold sample exists** — every single-stream window logged 94–97% prefix-cache hit; best cache-contaminated sample 6,946 tok/s, so the true cold rate can only be *lower* |
| Decode, single stream | 80–94 tok/s (DSPARK) | p50 80.7 / p90 101.3 / max 176.7 tok/s (n=234) |
| Prefix-cache benefit | 0.34 s ≈ 325K tok/s (39×) | 43–57K tok/s at ~96% hit |
| Footprint | TP8, ~64 GB/card (769 B-class MoE) | TP8, ~38 GB/card (306 GB weights) |

**Correction to a widely repeated figure:** the `75,647 tok/s` line attributed to GLM-5.3 is `Running: 5 reqs … Prefix cache hit rate: 96.5%` — a *cache-assisted, multi-request* number, not compute throughput. It belongs in the cache-benefit row, not the prefill row, and it is still ~5× below this model's cache-hit rate. Any "GLM prefill ≈ 9K tok/s" claim is unsupported by the logs.

**Net:** compute throughput is the same order of magnitude on both; single-stream decode is a tie; DSV4.1-Flash has the larger cache-reuse payoff and the much larger model.

## 6. Architecture limits found (do not re-litigate)

| Attempt | Result |
|---|---|
| DP2 × TP4 (any EP size) | **Rejected by the model**: `ValueError: V4.1 vision currently supports TP/EP without DP, CP, PP or MoE A2A` — both DP replicas fail within ~70 s |
| Prefill context-parallel (`prefill_cp`) | Same class of rejection |
| `--enable-dp-attention` alone | Silently builds no DP group (`dp_size: 1`) — looks like it works, isn't |
| DSPARK + DP attention | Requires `--enable-dp-lm-head` (and is unreachable anyway, per row 1) |
| **TP8/EP8 (shipped)** | Only layout that starts and serves |

Additional SM120 gaps confirmed: MSCCL++ (`_tune` kernel crashes), FlashInfer all-reduce fusion, `flashmla_sparse_q8`, and CUDA-graph capture for **prefill** (crashes even with DSPARK off — unrelated to speculation).

SGLang-specific configuration traps: `--enable-expert-parallel` is a *vLLM* flag (SGLang uses `ep_size`); `docker run` must pass `TP_SIZE`/`DP_SIZE`/`EP_SIZE` explicitly or the container silently serves TP8 defaults; and `ep_size ≤ tp_size` or the MoE rank computation divides by zero.

## 7. Operational lessons (cost us real experiments)

1. **A stopped container can still block a restart, and a stale one can still answer.** A defunct container often survives a single `docker rm -f`; the next `docker run` then fails with a name conflict, while an orphaned engine process keeps port 8000 and happily answers health checks. Always loop: `docker rm -f` until `docker ps -a` no longer lists the container **and** nothing is listening on the port. Concrete symptom we hit: a leg labelled `spec_none` reported `speculative_algorithm=DSPARK, block_size=12` — it was measuring the previous leg.
2. **Verify the config you intended, from the server, before timing anything.** `/get_server_info` for `speculative_algorithm`/`block_size` plus a `/v1/models` 200 is the minimum gate. A leg that fails the gate must be *discarded*, not recorded — a mislabelled number is worse than a missing one.
3. **A parse failure is not evidence.** An unparseable response body must not satisfy any "expected value" assertion (that is how a not-yet-started server got recorded as "spec off").
4. **Long tasks die with the SSH session.** `nohup`/`setsid` across two SSH hops was killed three times before we switched to `at`; every multi-minute step (weight load, sweeps, restores) now runs as an `at` job with a log file.
5. **Restore the shipped config automatically at the end of every sweep** — the restore leg is a step in the sweep, not an afterthought, and it must be verified by the same gate as the experiment legs.
6. **Never count streaming chunks as tokens.** Use `stream_options.include_usage`; chunk-counting under-reported decode by ~2×.
7. Host-network containers need `--noproxy '*'` when calling `localhost`.

## 8. What we did *not* establish

- No 24-hour soak test: the numbers above come from a working day of production use with experiment-driven restarts, not an unattended endurance run.
- Block sizes 4/8/12 remain unmeasured (§4.4) — the sweep tool was fixed afterwards; re-run before quoting.
- Long-context degradation beyond 32K prefill was not characterised; only the 128K cold point and the 1M context *limit* are known.
- All figures are single-user, single-stream, no batch concurrency target.

## 9. Reproduction artifacts

| Artifact | Location |
|---|---|
| Launcher (env pass-through, production defaults) | `/ssd/dsv41/run_dsv41.sh`, `boot.py` |
| Sweep with per-leg config verification | `/ssd/dsv41/bench/sweep_v3.sh` (logs: `/ssd/dsv41/state/sweep_v3.log`, `.jsonl`) |
| Production restore (used after every experiment) | `/ssd/dsv41/restore_prod.sh` |
| Prefill sweep driver | `/ssd/dsv41/benchmarks/ssbench4.py` |
| Client-path benchmark | `bench_live_model.py` (Mac side, exercises the tunnel) |
| Raw logs | `/ssd/dsv41/exp_prefill_20260923.log`, `/ssd/dsv41/exp_round2_20260923.log` |

## Corrections to prior notes

- **Status changed**: the 2026-09-22 note ("starts and serves, but runtime-crashes → not production-ready") no longer describes the deployment. With `chunked_prefill_size=2048`, `DSPARK block_size=6`, TP8/EP8 and prefill CUDA graph **disabled**, the server ran a full working day as the production backend; observed restarts were experiment-induced, not spontaneous.
- **`max_running_requests` is 16**, not 8 (see §2).
- **GLM-5.3 "decode ~100 tok/s, prefill ~9K tok/s" is not supported by its own logs** — see §5. The defensible figures are p50 80.7 tok/s decode and no measurable cold prefill.

---

*Measured 2026-09-23 on the deployment described above. Every number in §4–§5 is reproducible from the artifacts in §9.*