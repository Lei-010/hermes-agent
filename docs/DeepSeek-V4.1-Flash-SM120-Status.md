# DeepSeek-V4.1-Flash on RTX PRO 6000 Blackwell (SM120) — Status & Working Config

> Status: **Starts and serves, but runtime-crashes** — not production-ready as of 2026-09-22.
> Contrast: [GLM-5.3-Flash deployment guide](GLM-5.3-Flash-SM120-Deployment.md) (production-ready on same hardware).

## Hardware

8× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB, sm_120, CC 12.0, driver 590.48.01)

## Configuration that reached serving state (SGLang, TP=8)

```bash
docker run --gpus all -d --name dsv41 \
  --shm-size=64g --ipc=host --network host \
  --restart unless-stopped \
  -v /ssd/DeepSeek-V4.1-Flash:/model \
  deepseek-v41-4x6000:local \
  python3 -m sglang.launch_server \
    --model-path /model \
    --tp 8 \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code
```

### Key server_args

| Parameter | Value |
|-----------|-------|
| tp_size / ep_size | 8 / 8 |
| attention_backend | `dsv4` |
| kv_cache_dtype | `fp8_e4m3` |
| mem_fraction_static | 0.72 |
| max_running_requests | 8 |
| context_length | 1,048,576 |
| speculative_algorithm | DSPARK (block_size=6) |
| moe_runner_backend | `flashinfer_mxfp4` |
| CUDA graph decode | full, max_bs=8 |
| CUDA graph prefill | disabled |

## Failure mode

Server reaches `"fired up and ready to roll!"` and serves requests briefly, then crashes at runtime with `Connection refused` on internal health-check → container restart loop. Root cause likely one of: MHC kernel gap, Lightning Indexer SM120 integration, or memory pressure under serving load.

## Known issues affecting SM120 (as of 2026-09-22)

| Issue | Status | Timeline |
|-------|--------|----------|
| MHC kernel (DeepGEMM) | PR #447 open (741 tests pass) | Weeks |
| Lightning Indexer | **flashinfer PR #5075+5197+5226 merged** | ✅ Fixed (flashinfer level) |
| HiRadix | Stale, no fix | Unknown |
| FP8 KV scale bug | vLLM #56892 + PRs #57028/#57292 in flight | 1-2 months |
| MXFP4 → Marlin fallback | sglang #38685/#38170 in flight | Weeks |
| FP8 SM120 checkpoint | NVIDIA NVFP4 released 9-17 | ✅ Available |

## Contrast: GLM-5.3-Flash on same hardware

GLM-5.3-Flash (321B/18B, NoPE MLA, FP8, 306 GB) runs **stably** with vLLM TP=8 via community patch: decode ~100 tok/s, prefill ~9K tok/s, 1M context all-pass. See [GLM-5.3-Flash-SM120-Deployment.md](GLM-5.3-Flash-SM120-Deployment.md).

The key difference: GLM uses KDA linear attention (structurally reduces KV 4.44×) with a single MLA variant — one integration gap, now fixed. DSV4.1-Flash stacks five novel mechanisms (Engram memory, causal encoder-decoder, DSPARK, dual-tier sparse attention, Hyper-Connections) — each requiring separate SM120 kernel support.

## Recommendation

| Scenario | Recommendation |
|----------|---------------|
| gpu01主力 | GLM-5.3-Flash (stable, faster) |
| DeepSeek需求 | deepseek-v4-pro cloud API |
| DSV4.1-Flash本地 | Wait 3-6 months for SGLang maturity |
| 急用本地DeepSeek | V4-Flash-0731 (200 GB, officially verified on RTX PRO 6000) |

## Tracking

- vLLM #56700 (TP8 experience) · #56892 (FP8 KV bug) · #54929 (original report)
- sglang #23602 (V4 roadmap) · #26690 (HiRadix) · #24692 (SM120 kernel stack)
- DeepGEMM #317 (closed) → PR #447 (SM120 support)
- flashinfer PR #5075, #5197, #5226 (all merged)
- NVIDIA NVFP4 checkpoint: [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)

---

*Last updated: 2026-09-22*
