# GLM-5.3-Flash on NVIDIA RTX PRO 6000 Blackwell (SM 120) — 部署与运行指南

> 在 8×RTX PRO 6000 Blackwell（sm_120）上运行 GLM-5.3-Flash（320B-A18B MoE，原生多模态，MIT 许可）
> 的完整配置记录。官方 vLLM 无法直接在 SM120 上启动 GLM-5.3-Flash（NoPE MLA 架构限制），
> 本文档记录根因分析与已验证的解决方案。

## 硬件与模型

| 项 | 配置 |
|----|------|
| GPU | 8× NVIDIA RTX PRO 6000 Blackwell（sm_120，96GB GDDR7/卡） |
| 模型 | GLM-5.3-Flash（智谱，320B 总参 / 18B 激活 MoE，FP8 原生权重 ~306GB） |
| 推理引擎 | vLLM（社区特制镜像 `vllm/vllm-openai:glm53-sm120nope`） |
| 服务形态 | OpenAI 兼容 API（vLLM serve），TP=8 |

## 问题：官方 vLLM 在 SM120 上启动失败

### 根因

GLM-5.3-Flash 采用 **NoPE MLA 架构**（`qk_rope_head_dim=0`，无位置编码的 Multi-head
Latent Attention）。vLLM 中唯一支持 SM120 的 MLA 后端为
`FLASHINFER_MLA_SPARSE_SM120`，该后端**强制使用 fp8_ds_mla 缓存格式**，其 kernel
硬编码 `pe_dim=64`（DeepSeek MLA 参数）——GLM 的 NoPE 变体 `pe_dim=0` 直接触发
断言失败，无法启动。

### 解决方案：社区补丁移植

来源：[tonyd2wild/DGX-Spark](https://github.com/tonyd2wild/DGX-Spark) 仓库的
`Dockerfile.glm53-sm121`（纯 Python 文本替换，架构无关，可移植到任意 SM120 机器）。

**补丁要点**：

1. `cuda.py`：SM120（compute capability 12.0）候选后端追加 SM90 系列，能力门控
   从 `(9, 0)` 扩展到 `(9, 12)`
2. KV cache 格式改为 bf16，绕开 fp8_ds_mla 的 `pe_dim=64` 硬编码检查
3. FlashInfer 0.6.18 以 Python 源码 + CUDA cubin 双包内置（避免在线编译）

## 镜像与启动

### 镜像

```
vllm/vllm-openai:glm53-sm120nope
```

（官方 vLLM OpenAI 服务器 + FlashInfer 0.6.18 + NoPE prefill 补丁；
2026-09-14 合入 vLLM 主线的 NoPE prefill 修复 PR #55738 已包含在该镜像中）

### 启动脚本（v21，已验证稳定）

```bash
#!/bin/bash
# /ssd/serve_glm53_v21.sh — GLM-5.3-Flash on 8×RTX PRO 6000 Blackwell (sm_120)
MODEL="/ssd/GLM-5.3-Flash"

docker run --gpus all -d --name glm53 \
  --shm-size=64g --ipc=host --network host \
  --restart unless-stopped \
  -v $MODEL:/model \
  -e FLASHINFER_MLA_SPARSE_SM90=1 \
  vllm/vllm-openai:glm53-sm120nope \
  --model /model \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --max-model-len 1048576 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.85 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --host 0.0.0.0 --port 8001
```

**关键参数说明**：

| 参数 | 值 | 理由 |
|------|-----|------|
| `FLASHINFER_MLA_SPARSE_SM90=1` | 环境变量 | 钉死 MLA 后端，防止回退到不支持的 SM120 路径导致 crash |
| `--tensor-parallel-size 8` | TP8 | 306GB 权重分摊 8 卡，约 42GB/卡 |
| `--max-model-len 1048576` | 1M | 原生 1M 上下文 |
| `--max-num-seqs 512` | 512 | Mamba 缓存块限制（1024 会 OOM 崩溃） |
| `--gpu-memory-utilization 0.85` | 0.85 | 留出 CUDA graph 与激活空间 |
| `--quantization fp8` | FP8 | 原生 FP8 权重，免转换 |

## 已验证性能

| 指标 | 数值 |
|------|------|
| decode 吞吐 | ~100 tok/s（CUDA graph 开启后，eager 模式仅 9.3 tok/s，加速 10.4×） |
| prefill 吞吐 | ~9000 tok/s（1M 上下文时 7.7K tok/s） |
| TTFT（万字 prompt） | 1.74s |
| CUDA graph | FULL 51 张图 |
| KV cache 池 | 3.03M tokens（并发 2.89×） |
| 长上下文大海捞针 | 128K / 512K / 1M 三档两针全中 |

## 网络拓扑（远程访问）

```
Mac (localhost:8101)
  ← ssh -L 8101:localhost:8001 -J digs01 gpu01   (SSH 隧道，LaunchAgent 常驻)
    → gpu01:8001 (vLLM API)
```

- gpu01 无外网 → 出网走 Mac 反向 SOCKS 代理（端口 11080）
- Hermes Agent 通过 `provider: local-glm53, base_url: http://127.0.0.1:8101/v1` 接入

## 踩坑记录

| # | 问题 | 结论 |
|---|------|------|
| 1 | 升级 FlashInfer 解决 SM120 crash | ❌ 无效——crash kernel 在 vLLM 自带 `cache_kernels.cu`，非 FlashInfer |
| 2 | DSV4 与 GLM 同时满载 | ❌ 显存互斥——8 卡只能跑一个，切换用 `/ssd/dsv4_start.sh` |
| 3 | GLM-5.3 思考模式 | 小 `max_tokens` 会被 reasoning 吃光导致 content 为空——**非故障**，调大即可 |
| 4 | Mac 本地代理 8888 劫持 localhost | 访问隧道端口须 `--noproxy '*'` |
| 5 | 上游 NoPE 修复跟踪 | vLLM issue #53963 |

## 与 DeepSeek-V4.1-Flash 的共存

同一台 gpu01 上 DeepSeek-V4.1-Flash（SGLang，TP=4/8）与 GLM-5.3-Flash（vLLM，TP=8）
**显存互斥，不可同时满载**。切换：

```bash
docker stop glm53 && bash /ssd/dsv4_start.sh    # 切到 DS
docker stop dsv41 && bash /ssd/serve_glm53_v21.sh  # 切回 GLM
```

## 致谢

- [tonyd2wild/DGX-Spark](https://github.com/tonyd2wild/DGX-Spark) — SM120 NoPE 补丁原始来源
- [vLLM PR #55738](https://github.com/vllm-project/vllm/pull/55738) — 主线 NoPE prefill 修复
- 智谱 AI — GLM-5.3-Flash 开源（MIT）

---

*部署验证日期：2026-09-20 · 环境：Rocky/CentOS + NVIDIA driver 595.x + CUDA 13.x*
