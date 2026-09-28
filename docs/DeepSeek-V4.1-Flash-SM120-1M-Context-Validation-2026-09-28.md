# DeepSeek-V4.1-Flash on SM120 — 1M Context Validation (Speed & Stability, 2026-09-28)

> Scope: validate the running production container with **`CONTEXT_LENGTH=1048576`**
> (4× RTX PRO 6000 Blackwell Max-Q / SM120, TP4-EP4, DSPARK speculation) for
> **prefill speed, decode speed at extreme context, and stability**.
> Supersedes the 409,600-token deployment snapshot in
> [Live Configuration & Concurrency Benchmark (2026-09-27)](DeepSeek-V4.1-Flash-SM120-Live-Config-and-Concurrency-2026-09-27.md):
> production has since been re-created with the 1M context budget (2026-09-27 22:24 UTC).
>
> Everything below is measured on the live endpoint 2026-09-28 08:22–09:04 CST;
> three of the numbers correct earlier statements in our own notes and in a public issue thread (§7).

---

## 1. Container under test

| Item | Value |
|---|---|
| Container | `dsv41`, id `5b66bd6557b5`, created 2026-09-27T22:24:17Z |
| `CONTEXT_LENGTH` | **1048576** |
| `max_req_input_len` (server) | **1048570** |
| KV pool (`max_total_num_tokens`) | 1,137,920 |
| `max_running_requests` / decode graph | 8 / 8 |
| Speculation | DSPARK, block size 6 |
| KV dtype / `mem_fraction_static` | `fp8_e4m3` / 0.82 |
| GPU memory (steady, after tests) | 96,817–96,839 MiB of 97,887 MiB per card (**98.9 %**) |
| `RestartCount` | 0 for the entire test window |

---

## 2. Method — the unique-prefix rule

Everything here was sent with a **unique marker at the head of every prompt**
(`[doc-u<purpose>-<HHMMSS>] …`). This is not cosmetic:

- sglang's radix cache keeps a completed request's pages resident.
- A follow-up request that shares its prefix skips prefill for the shared part, so
  "1M tokens in 75 s" can be measured while almost nothing is prefilled.
- Two earlier data points (ours and a parallel session's) were exactly that artifact — see §7.

Requests are non-streaming for the ladder (only `usage.prompt_tokens` matters) and streaming
with `stream_options.include_usage` for the decode probes (to get TTFT and a server-side
completion count). Token counts are always the server's `usage`, never chunk counts
(DSPARK packs 2–5 tokens per SSE chunk).

Artifacts on the inference host: `/ssd/dsv41/ab/ladder_cold.py`,
`probe_hi_ctx.py`, `stability.py`, plus `/ssd/dsv41/ab/run_dsv41_400k.sh` (one-command rollback
to the 400K budget).

---

## 3. Cold prefill (unique prefix ⇒ genuinely cold)

| Prompt tokens | Elapsed | Prefill rate |
|---|---|---|
| 402,625 | 79.5 s | 5,065 tok/s |
| 704,582 | 175.2 s | 4,022 tok/s |
| 905,885 | 260.5 s | 3,478 tok/s |
| **1,006,538** | **310.5 s** | **3,242 tok/s** |
| **1,006,538** (repeat, new unique prefix) | **322.0 s** | **3,126 tok/s** |

- **A full 1M-token cold prefill works, twice**, with a healthy completion (`"OK"`) each time.
- Rate declines monotonically with length (5,065 → 3,126 tok/s); attention cost per token grows
  while the fixed per-request cost amortises.
- Practical ceiling for interactive use: **~5 minutes** to cold-fill 1M tokens on this box.

---

## 4. Decode at short context (256 generated tokens, temp 0)

| Concurrency | Aggregate | Per-stream | TTFT | TPOT |
|---|---|---|---|---|
| 1 | 152.5 / 120.2 tok/s | 164.8 / 128.7 | 0.125 s | **6.1 ms** |
| 2 | 229.1 tok/s | 162.3 | 0.149 s | 6.6 ms |
| 4 | 302.0 tok/s | 98.2 | 0.148 s | 10.6 ms |
| 8 | 452.0 tok/s | 94.9 | 0.155 s | 10.7 ms |

4.7K-token prefill: TTFT 0.791 s. These match the 409,600-token deployment almost exactly
(single stream 152 vs 153.8 tok/s) — **raising `context_length` to 1M did not cost short-context
decode speed** on this build, contrary to the ~2.68× "decode scan width" lever reported for
SM90 (that lever did not materialise here).

---

## 5. Decode at ~906,000 tokens of context

Same 906K-token prefix, one cold fill then repeated requests (prefix cache hits):

| Run | TTFT | Decode | TPOT | Completion |
|---|---|---|---|---|
| REQ1 cold fill | 260.5 s | — | — | 2 tokens |
| REQ2 warm prefix | **2.38 s** | 129.3 tok/s | 8 ms | 80 tokens |
| REQ3 warm prefix | **2.39 s** | 130.1 tok/s | 8 ms | 80 tokens |
| SEQ1 (fresh instruction) | 260.4 s (cold again) | 92.2 tok/s | 11 ms | 48 tokens |
| SEQ2–SEQ5 (cached prefix) | 2.37–2.40 s | 99.8–100.8 tok/s | 10 ms | 48 tokens each |

- **Decoding with ~906K tokens in context runs at ~100 tok/s (129–130 for short outputs) with
  TPOT 8–10 ms** — within ~1.5× of the short-context rate. Extreme context does not collapse
  decode on this deployment.
- **Variance across five sequential repeats is below 1 %** (TTFT 2.37–2.40 s).
- **Prefix caching is the whole game for interactive use: 260 s → 2.38 s (109×)** at 906K tokens.
- Note SEQ1: after a gap of some minutes (other traffic), the 906K prefix had been evicted and
  the cost returned to 260 s. Long-context caching is opportunistic, not guaranteed — plan for
  re-prefill if other traffic fills the pool.

### Concurrency at extreme context degrades hard

4 concurrent requests over the same 906K prefix, 64 tokens each:

| Metric | Value |
|---|---|
| Wall | 8.6 s for 256 generated tokens |
| **Aggregate decode** | **29.7 tok/s** (per-stream ≈ 7.4) |
| TTFT (all four) | 7.5–7.9 s |

One long-context stream achieves ~100 tok/s; four of them together achieve ~30 tok/s in
aggregate. Attention over 4×906K tokens per step is the cost, and `max_running_requests=8` is
nominal — **at long context this endpoint serves one heavy request well, not several.**

---

## 6. Stability

- Two independent cold 1M prefills (310.5 s / 322.0 s) — both completed, correct output.
- Repeated high-context decode: 5 sequential + 4 concurrent, no failures.
- No traceback, no OOM, no CUDA error in the container log for the whole window.
- `dmesg` Xid count **constant at 52** before and after every test (no new Xid).
- `RestartCount = 0`; GPU memory steady at ~96.8 GB / 97.9 GB (98.9 %), with ~94.9 GB observed
  during the 1M prefill — the pool is nearly fully committed, so headroom for an extra
  long-context request is ~1 GB, not gigabytes.

Risks that remain:

1. **Fresh-start big request.** See §7.3 — the one crash we observed happened minutes after a
   cold start, on the first oversized request.
2. **No memory headroom.** At 98.9 % occupancy, a second very large request competes for a pool
   that is effectively sized for one.
3. **Cold-start latency is ~5 min** for a full 1M fill; any workload that changes the prefix
   every turn pays it in full.

---

## 7. Corrections

### 7.1 Our own 2026-09-27 "409K cold = 62.7 s (6,528 tok/s)" was not cold

That request reused the prefix of the immediately preceding calibration request. Re-measured
with a unique prefix: **402,625 tokens = 79.5 s (5,065 tok/s)**. The earlier figure overstated
cold prefill by ~29 %.

### 7.2 A parallel session's "1040K = 74.5 s (13,962 tok/s)" is a cache hit, not a cold prefill

From the same host's ladder log:

| Level | prompt_tokens | Elapsed | Implied rate |
|---|---|---|---|
| 400K | 399,890 | 75.9 s | 5,267 tok/s |
| 900K | 899,744 | 179.1 s | 5,025 tok/s |
| 1040K | 1,039,704 | **74.5 s** | **13,962 tok/s** |

1.04M tokens at nearly 3× the rate of the 900K level is not physically reachable on this
hardware; the request consumed the 900K prefix already resident in the radix cache and only
prefilled the remaining ~140K tokens (≈1.9K tok/s of new work, consistent with a loaded pool).
The capability claim (a 1.04M-token request is accepted and answered) is **correct**; the
throughput figure must not be quoted. Our matching cold measurement is in §3.

Consequence to keep in mind: a routing/gating script that only greps for
`RESULT[<size>]: prompt_tokens=` will pass on a cache hit. Force a unique prefix per level.

### 7.3 The 2026-09-27 23:5x SIGKILL of a ~500K request did **not** reproduce

That crash: whole TP group died during the next large prefill, `Exited (137)`, no Python
traceback, no CUDA error, no OOM-killer line, host RAM abundant.

Today, on the same configuration and after warm-up: cold 500K (106.1 s), then cold 1M twice,
plus 5 sequential and 4 concurrent 906K-context decodes — all fine.

Best available explanation (⚠️ hypothesis, not proven): the first oversized request after a
cold start triggered **new Triton kernel-shape JIT compilation** while the engine was already
under a big prefill, stalling the scheduler past sglang's watchdog (300 s) → SIGKILL of the
process group, which leaves no Python traceback. Consistent log evidence captured at the time:

```
Triton kernel '_hc_mix_reduce_sinkhorn_kernel' took 20.78 s to compile after serving started.
Serving-time compilation can stall the engine; pre-compile it during engine init.
Health check failed. Server couldn't get a response from detokenizer for last 20 seconds.
```

and today's 40-minute window contains **no** compile or watchdog message.

**Operational rule suggested by this**: after a cold start of the 1M configuration, send one
"warm-up" request of a few hundred K tokens and wait for it to complete before relying on the
service. The hypothesis is falsifiable: restart, then issue a cold ≥500K request as the very
first request, and see whether the group dies again.

---

## 8. Guidance for running 1M in production

- Keep `CONTEXT_LENGTH=1048576` only if the workload genuinely needs >400K; the cost is a
  ~5-minute cold fill and a pool with ~1 GB of slack.
- **Never benchmark without a unique prefix** (§2, §7.2), and never quote a warm number as cold.
- Prefer workloads with a **stable long prefix** (that is what turns 260 s into 2.4 s).
- Treat the endpoint as **one heavy request at a time**: 4-way long-context concurrency ≈ 30 tok/s
  aggregate.
- If a crash recurs, capture `docker inspect` + last 300 log lines before restarting —
  `/ssd/dsv41/ab/` keeps the scripts used here.

---

## 9. Open items

1. Falsify/confirm §7.3 by making a cold ≥500K request the first request after a restart.
2. Decide whether to raise `MEMORY_FRACTION` (more KV slack, less activation headroom) or lower
   `MAX_RUNNING_REQUESTS` to reflect the real long-context concurrency limit.
3. Re-check `Xid=52` (present since before this test window, unchanged by any test here) — its
   originating events predate this work.
4. If publishable, the cache-hit artifact in §7.2 is worth a short note on the upstream issue
   thread where the same ladder was quoted.

---

*Measured 2026-09-28 08:22–09:04 CST on `dsv41` (container 5b66bd6557b5) by an automated
Hermes session; raw logs in `/ssd/dsv41/ab/{ladder,probe,hictx,stab}_0928_*.log`.*