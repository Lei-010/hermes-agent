# DeepSeek-V4.1-Flash on RTX PRO 6000 (SM120) — Live Configuration, Runtime Status & Concurrency Benchmark

> Status: **In production** as the local inference backend of a single-user Hermes deployment.
> Snapshot taken **2026-09-27** (host clock: CST / UTC+8, `gpu01`).
> ⚠️ **Context budget superseded (2026-09-28)** — production has since been re-created with
> `CONTEXT_LENGTH=1048576`; see [1M Context Validation (2026-09-28)](DeepSeek-V4.1-Flash-SM120-1M-Context-Validation-2026-09-28.md) for the verified 1M numbers, the unique-prefix
> measurement rule, and three corrections to earlier figures (including one in this document).
> Supersedes the *deployment description* in
> [DeepSeek-V4.1-Flash-SM120-Performance-and-Production-Config.md](DeepSeek-V4.1-Flash-SM120-Performance-and-Production-Config.md) (2026-09-23),
> which was measured on an **8-GPU TP8/EP8** layout that is **no longer what runs**. See §5 for the delta.
> Related: [GLM-5.3-Flash deployment guide](GLM-5.3-Flash-SM120-Deployment.md) ·
> [DeepSeek-V4.1-Flash SM120 status](DeepSeek-V4.1-Flash-SM120-Status.md) (2026-09-22, historical).

Every number below is either read from the live host (`docker inspect`, `nvidia-smi`, `/state/launch.json`)
or measured against the live endpoint on 2026-09-27 with the method in §6. Nothing is estimated.

---

## 1. Hardware and topology

| Item | Observed value |
|---|---|
| Host | `gpu01`, up 3 days 11 h at snapshot time |
| GPUs present | **7×** NVIDIA RTX PRO 6000 Blackwell **Max-Q Workstation Edition**, 97,887 MiB each |
| Driver / CUDA | 590.48.01 / CUDA 13.1 |
| Interconnect | PCIe, **no NVLink** |
| GPUs in use | **4** — container sees devices `0–3` only; `4–6` idle (0 MiB, 4–15 W) |
| Per-GPU VRAM in use | ≈ **86.1 GB / 97.9 GB** per active card |

⚠️ **Accounting note**: the host exposes **7** devices (`nvidia-smi -L`), while the earlier
2026-09-23 note describes **8** GPUs. This document records the observed count (7) and does not
explain the difference — confirm before citing either figure as a hardware spec.

---

## 2. Serving configuration (what actually runs)

### 2.1 Container image

| Item | Value |
|---|---|
| Image | `deepseek-v41-4x6000:local` |
| Image ID | `sha256:adbfd51721e1576d3f8a19b88bec6ac4af057883dfbd595e380fa4716a8f9a27` |
| Image size / built | 35,903,089,108 B (**35.9 GB**) |
| Container | `dsv41`, `--network host`, restart policy `unless-stopped` |
| Entrypoint | `python3 -u /opt/dsv41/boot.py` |
| Launch record | `/state/launch.json` (written by `boot.py`) |
| Model weights | `/ssd/DeepSeek-V4.1-Flash` — **476 GB**, 48 shards, revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d` |

⚠️ **Provenance is not reproducible from the image itself**: its labels carry
`ai.sglang.image.tag=local/sglang:dev`, `ai.sglang.build.commit=unknown`,
`ai.radixark.overlay.commit=unknown`. `pip show sglang` inside the container reports `0.0.0.dev0`
(a source checkout, no version stamp). The image is an overlay (extra layers replace
`srt/models/deepseek_v4.py` and `kernels/ops/attention/flash_mla_sm120.py`, add a compiled
`librow_store.so` and the `boot.py` supervisor).

⚠️ It is **not** `lmsysorg/sglang:v0.5.20`: layer-by-layer comparison shows the local image (87 layers)
diverges from `lmsysorg/sglang:v0.5.20` (71 layers) at layer 17 and is not a prefix-superset of it,
nor of `lmsysorg/sglang:dev-dsv41` (70 layers). The SM120 DeepGEMM paged-MQA code paths are present in
the container, but **the baseline commit they came from cannot be established** — do not claim this
image equals any released tag.

### 2.2 Launch arguments (verbatim, `/state/launch.json`)

| Flag | Value |
|---|---|
| `--served-model-name` | `deepseek-v4.1-flash` |
| `--tp` / `--ep-size` | **4 / 4** |
| `--attention-backend` | `dsv4` |
| `--moe-runner-backend` | `flashinfer_mxfp4` |
| `--mem-fraction-static` | `0.82` |
| `--context-length` | **409600** |
| `--max-running-requests` | **8** |
| `--cuda-graph-max-bs-decode` | **8** |
| `--chunked-prefill-size` | 2048 |
| `--speculative-algorithm` / block size | **DSPARK** / 6 |
| `--tool-call-parser` / `--reasoning-parser` | `deepseekv41` / `deepseek-v4` |
| `--load-format` | `safetensors` |
| `--min-free-slots-delay` / `--random-seed` | 1 / 0 |
| `--skip-server-warmup` | enabled |
| `--host` / `--port` | `0.0.0.0` / **8000** |
| KV-cache dtype (from the server-args dump) | `fp8_e4m3` |
| Offload mode (recorded by `boot.py`) | **`ram`** |

**Serving lane**: the server is reached from the operator's Mac through
`ssh -L 8102:localhost:8000 -J <jump host> <gpu01>`; the client-side Hermes profile points at
`http://127.0.0.1:8102/v1`. Note the client config declares `context_length: 1048576`, while the
server's real ceiling is **409,600** — the server value is authoritative.

---

## 3. Runtime status at snapshot

| Item | Value |
|---|---|
| Container started | 2026-09-27 13:51:32 UTC (= 21:51 CST), `RestartCount = 0` |
| Uptime at snapshot | ≈ 1 h |
| GPU 0–3 | 167–200 W, 38–45 °C, ≈86.1 GB VRAM, idle at the moment of sampling |
| GPU 4–6 | idle — 0 MiB, 4–15 W, 19–22 °C |
| `/v1/models` | `deepseek-v4.1-flash`, `max_model_len = 409600` — responding |

⚠️ `RestartCount = 0` with several distinct `StartedAt` values across 09-26 → 09-27 means the
container was **stopped and started manually, not crash-looped by the restart policy**. Restarts
observed in this window correlated with operator activity (benchmarks / handing GPUs back), not with
spontaneous failure — consistent with the conclusion in the 2026-09-23 note.

---

## 4. Measured performance — 2026-09-27

### 4.1 Decode throughput vs. concurrency

256 output tokens per request, `temperature=0`, `ignore_eos=true`,
`stream_options.include_usage=true`, 2 rounds per level (best aggregate reported).
Short prompt (44 prompt tokens).

| Concurrency | Aggregate | Scaling vs. 1 | Per-stream (median) | Requested TTFT (median) | TPOT (median) |
|---|---|---|---|---|---|
| 1 | **115.2 tok/s** | 1.00× | **121.9 tok/s** | 0.121 s | **8.2 ms** |
| 2 | **220.6 tok/s** | 1.91× | 125.8 tok/s | 0.147 s | 8.0 ms |
| 4 | **322.2 tok/s** | 2.80× | 108.5 tok/s | 0.643 s | 9.3 ms |
| 8 | **468.2 tok/s** | **4.06×** | 66.2 tok/s | 0.157 s | 15.2 ms |

Observations:

- **Concurrency 2 is effectively free** — per-stream rate is unchanged (125.8 vs 121.9 tok/s) while
  aggregate doubles.
- **8 concurrent streams is the hard ceiling** (`max_running_requests=8`, with the decode CUDA graph
  also compiled to batch size 8). At that level aggregate reaches 468 tok/s with per-stream still at
  66 tok/s (−46% vs. single stream).
- Beyond 8 the server queues (FCFS) rather than degrading gracefully; `retraction_policy=length`.

### 4.2 Prefill

| Case | Result |
|---|---|
| 4,714-token prompt, cold, streaming | TTFT **0.788 s** ⇒ ≈ **5,982 tok/s** |

Prefill remains communication-bound (PCIe, no NVLink), matching the earlier note's mechanism, though
the absolute rate is now lower than the 2026-09-23 figure because fewer GPUs participate (§5).

### 4.3 Measurement caveat worth reproducing

An earlier run of the same benchmark on **2026-09-26** (same container, same script) reported
**20–47 % lower** numbers (single stream 96.2 instead of 115.2 tok/s; 8-way aggregate 318.4 instead of
468.2). Cause: the operator's own interactive agent session was issuing requests against the same
8-request budget at the time, so the benchmark was queued behind live traffic. **Benchmark this
endpoint only when the interactive session is idle**, or explicitly budget for the shared cap —
otherwise the numbers understate the deployment by up to ~1.5×.

---

## 5. Delta vs. the 2026-09-23 document

| Dimension | 2026-09-23 note | **Live, 2026-09-27** |
|---|---|---|
| GPUs in use | 8 | **4** (host exposes 7) |
| Parallelism | TP8 / EP8 ("only layout that starts") | **TP4 / EP4** — starts and serves fine |
| `max_running_requests` | 16 | **8** |
| Image | (unnamed in that note) | `deepseek-v41-4x6000:local` (overlay, commit unknown) |
| Single-stream decode | prose 93.9 / structured 134.1 tok/s | **115–122 tok/s** (mixed prompt) |
| TTFT | 143–169 ms | **121–147 ms** |
| Prefill | ~7,000 tok/s | **~5,982 tok/s** |
| Concurrency / batching data | **none** ("single-user, single-stream") | §4 table above |

Reading of the delta (mechanism, not measurement): halving the tensor-parallel group halves the
cross-GPU communication per decode step, which **raises single-stream rate** while **lowering the
concurrency ceiling and prefill throughput**. The 2026-09-23 claim "TP8/EP8 is the only layout that
starts" describes that image; it does **not** hold for the current one.

---

## 6. Reproduction

Client endpoint: `http://127.0.0.1:8102/v1` (SSH tunnel to `gpu01:8000`), model `deepseek-v4.1-flash`.

Requirements for credible numbers:

1. **Count tokens from the server**, not from SSE chunk count —
   send `stream_options: {"include_usage": true}` and read `usage.completion_tokens`.
   With DSPARK speculative decoding a single chunk carries 2–5 tokens, so chunk counting
   under-reports throughput by 3–5×.
2. Set `ignore_eos: true` so every request generates the full `max_tokens` (otherwise the model stops
   early and the comparison is meaningless).
3. Run each level ≥2 rounds and report the best/median; discard runs that overlap interactive traffic.
4. Verify the interactive session is idle before trusting absolute values (§4.3).

Raw result JSON and the benchmark client used for this snapshot are the ones in §4; the method is a
plain streaming `POST /v1/chat/completions` with `temperature=0`.

---

## 7. Open items

1. **Image reproducibility** — the running overlay has no build-commit label; rebuilding it from a
   named SGLang tag is the prerequisite for citing this performance anywhere outside this deployment.
2. **4 vs. 7 vs. 8 GPUs** — host exposes 7 devices; only 4 are used by the container; the 2026-09-23
   note says 8. Reconcile before publishing a hardware spec.
3. **KV offload** — `offload_mode=ram` is in effect. Whether host-RAM KV paging contributes to the
   latency tail observed under contention (09-26) has not been isolated; `nvme` vs `ram` has not been
   A/B'd on this image.
4. **Concurrency headroom** — raising `max_running_requests`/`cuda_graph_max_bs_decode` above 8 requires
   freeing VRAM (≈86 GB of 97.9 GB per card is already committed, of which the KV pool is the
   adjustable part).

---

*Snapshot 2026-09-27 by an automated Hermes session on the operator's Mac; all figures re-read from the
host or re-measured against the live endpoint on that date.*