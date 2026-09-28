# GLM-5.3-Flash MTP Speculative Decoding on SM120 (RTX PRO 6000) — A/B Results & Production Config

> 2026-09-28 measured on 8× NVIDIA RTX PRO 6000 Blackwell (sm_120, 96GB each), TP=8,
> FP8 native checkpoint, community image `vllm/vllm-openai:glm53-sm120nope`.
> Follow-up to [GLM-5.3-Flash-SM120-Deployment.md](GLM-5.3-Flash-SM120-Deployment.md)
> (v21 baseline: no speculation, decode ~107 tok/s, prefill ~9K tok/s).

## TL;DR

MTP speculative decoding works on SM120 **with CUDA graphs** — the official recipe's
`enforce_eager: true` is a conservative default, not a hard requirement (source-verified:
only `deepseek_v32` is force-eagered in `config/speculative.py`). Final production config:

**spec3 (glm5_next_mtp) + CUDA graph + mem 0.97 + max-num-seqs 16 → decode 156 tok/s (+46% vs no-spec baseline), prefill 7.9K tok/s (unchanged), acceptance 2.1-2.4/3.**

## A/B matrix (single-stream decode 2K tokens streaming, 128K cold prefill ×2)

| Version | Config | decode tok/s | 128K prefill | acceptance | Verdict |
|---|---|---:|---:|---|---|
| v21 | no spec, mem0.91, seqs512, graph | 107 | 6.2-8.3K | — | old baseline |
| v3 | + spec3 **enforce_eager** | 19.5 ❌ | 5.8-7.6K | 2.35 | rejected: graph loss (10.4×) ≫ MTP gain (2.35×) |
| v4 | spec3 + graph, mem0.91, seqs512 | 133 ✅ | 7.7K | 2.3-2.9 | interim production |
| v5 | spec5 + mem0.93 + batch16k | 137 | 7.8K | 2.1-2.8/5 | rejected: +3% not worth 2× boot time (12 min) |
| v6 | spec3 + graph, mem0.97, **seqs4** | **165.1** | 6.1-8.0K | 2.5-3.0 | benchmark ceiling; concurrency capped at 4 |
| **v7** | spec3 + graph, mem0.97, **seqs16** | **156.4** | **7.9K** | 2.1-2.4 | **✅ production** |

Reference: krzychdre/GLM-5.3-Flash-sm120 (TP4, official FP8, MTP) reports 171.7 tok/s —
we reach 91% of that on TP8 (PCIe-only platform, TP8 all-reduce overhead).

## Pitfalls found (each cost one boot cycle)

1. **`disable_logprobs` is not a field** of `SpeculativeConfig` in this vLLM build —
   pydantic rejects the boot outright. Official older recipes include it; remove it.
2. **Method enum**: use `glm5_next_mtp` (model-specific; normalized internally to `mtp`).
   Generic `deepseek_mtp` also boots; `"vllm"` (from an early recipe draft) does not.
3. **FlashInfer version check**: `flashinfer-jit-cache (0.6.17) ≠ flashinfer (0.6.18.dev)`
   kills the registry subprocess at model-inspect. Must set
   `FLASHINFER_DISABLE_VERSION_CHECK=1` (the v21 script already had it; copy it).
4. **Boot trio** (copy verbatim from the known-good no-spec script):
   `FLASHINFER_DISABLE_VERSION_CHECK=1` + `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_SM90`
   + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
5. **`shm_broadcast: no block in 60s` during boot is normal** — it's the speculator
   graph capture (87 s at spec3, longer at spec5). GPU memory allocated + container
   Running + recovers in minutes = capture; GPU util 0% for tens of minutes = real hang.

## Why seqs (max-num-seqs) is the real decode lever

Cutting `max-num-seqs` shrinks captured CUDA graph batch sizes → smaller, faster graphs
→ single-stream decode jumps (133 → 165 from seqs512 → 4). But request #seqs+1 hard-queues.
Community benchmark configs (seqs=4) are single-stream-scored; **production seqs must be
set from your real concurrency profile, not copied from benchmark repos**. Ours peaks at
~8-12 concurrent (main agent + parallel subagents + 5 team profiles + cron), hence 16.

Mamba/hybrid note: `max-num-seqs 512` was originally set against Mamba cache-block limits
(1024 OOMs); smaller seqs *relieves* that pressure and frees KV-pool headroom.

## Production script (v7)

```bash
docker run -d --name glm53 --gpus all --privileged --ipc=host --shm-size=64g \
  --restart unless-stopped -p 8001:8000 \
  -v /ssd/GLM-5.3-Flash:/model \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_V1=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_SM90 \
  vllm/vllm-openai:glm53-sm120nope \
  --model /model --served-model-name glm-5.3-flash --trust-remote-code \
  --tensor-parallel-size 8 --quantization fp8 --max-model-len 1048576 \
  --max-num-seqs 16 --gpu-memory-utilization 0.97 --block-size 2304 \
  --no-enable-flashinfer-autotune --enable-auto-tool-choice \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --speculative-config '{"method":"glm5_next_mtp","num_speculative_tokens":3}'
```

No `--enforce-eager`. Graph capture succeeds ("Breakable CUDA graph enabled"); the draft
model runs the fallback rebuild path (not fused) — see next section.

## Known ceiling: fused multi-step draft decode (not unlockable by upgrading vLLM)

The fallback draft path is the gap between our 156 and theoretical ~230. Source-verified
(`v1/worker/gpu/spec_decode/autoregressive/speculator.py` L110-118): fused multi-step
draft requires the attention backend to implement `supports_draft_decode_metadata_update`;
only 4 backends do (flash_attn, mla/sparse_swa, mla/triton_mla, triton_attn). GLM's
backend trio (DEEPSEEK_V32_INDEXER / FLASHINFER_MLA_SPARSE_SM90 / KPOOL_TAIL) has none.
This is an upstream FlashInfer sparse-MLA feature gap — upgrading vLLM alone does not
unlock it. Re-evaluate when upstream adds the attribute.

## DFlash2 (block-diffusion draft) on SM120: one community success, ~30% odds for us

- samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell: same GPUs (SM120), C1 179.5 /
  C16 1045 tok/s, DFlash acceptance ~65% — but on a bespoke tpurtell vLLM fork, EXL3-K3
  quantized target (not official FP8), and CC BY-NC-ND draft weights (non-commercial).
- Mainstream SM120 deployments (krzychdre, yhfgyyf) all stay on MTP, not DFlash2.
- Verdict: blocked for production use (license + fork maintenance); revisit if Inco AI
  ships a commercially-licensed FP8 draft and the support lands upstream.

## Benchmark methodology

- decode: 35-token prompt, 2048 completion, `stream:true`, temp 0.7; rate =
  completion_tokens / (wall − 1 s). Server under otherwise-idle load.
- prefill: random-word prompts ~177k words (~206k tokens after tokenizer), 2 runs with
  different seeds (no prefix-cache hits), max_tokens=16.
- acceptance read from server `SpecDecoding metrics` logs (10 s cadence).
