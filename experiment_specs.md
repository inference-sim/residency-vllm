# Experiment Specifications

## Objective

Measure the latency overhead of per-tenant KV-cache residency instrumentation
in vLLM by running identical workloads against a patched server and a stock
server simultaneously.

---

## Hardware

| Resource | Specification |
|---|---|
| GPU | NVIDIA H100 80GB (1 per variant, 2 total for A/B) |
| GPU allocation | `nvidia.com/gpu: 1` per Kubernetes pod |
| Cluster | OpenShift 4.x on bare-metal GPU nodes |
| Storage | Shared PVCs (RWX) for model weights and results |

---

## Model

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-14B` |
| Parameters | 14B |
| Precision | Default (BF16) |
| Context length | Default (32k) |
| vLLM version (vanilla) | `vllm/vllm-openai:v0.23.0` |
| vLLM version (patched) | `ghcr.io/inference-sim/residency-vllm:0.23.0-residency-v2` |
| Server flags | `vllm serve --model Qwen/Qwen3-14B --port 8000` |

---

## Workload

| Parameter | Value |
|---|---|
| Driver | [blis observe](https://github.com/inference-sim/inference-sim) |
| Workload spec format | TraceV2 YAML (declarative) |
| Arrival process | Poisson (independent per tenant) |
| Prompt tokens | 1024 (blis calibrates word count to hit target) |
| Max output tokens | 128 (enforced via `min_tokens = max_tokens`) |
| Streaming | Yes (SSE) |
| Duration | 600 seconds per experiment |
| Random seed | 42 |
| Warmup | None (cold start included) |

### Token Distribution

The workload driver (`blis observe`) generates prompts calibrated to produce
exactly 1024 tokens when tokenized by the target model. It uses an internal
calibration step to determine the word-to-token ratio for the specific model's
tokenizer, then generates prompts of the appropriate word count.

| Property | Value |
|---|---|
| Target input tokens | 1024 |
| Chat-template overhead | ~8 tokens (ChatML format) |
| Total tokens seen by model | ~1032 |
| Output tokens per request | 128 (enforced via `min_tokens = max_tokens`) |

All tenants use identical token distributions (homogeneous workload). The only
difference between tenants is their arrival process timing (seeded by the
workload spec).

### Tenant ID injection

Each request includes the tenant ID via HTTP header:

```
x-gateway-inference-fairness-id: tenant_A
```

The patched server reads this header in `serving.py` and stores it in
`sampling_params.extra_args["tenant_id"]` for residency accounting. The vanilla
server ignores the header.

---

## Experiment Matrix

### Rate Sweep

Fixed: 2 tenants (`tenant_A`, `tenant_B`), 600s duration, seed 42.

| Experiment | Per-tenant rate | Aggregate rate | Total requests (approx) |
|---|---|---|---|
| `agg_2rps` | 1 req/s | 2 req/s | ~1,200 |
| `agg_4rps` | 2 req/s | 4 req/s | ~2,400 |
| `agg_6rps` | 3 req/s | 6 req/s | ~3,600 |
| `agg_8rps` | 4 req/s | 8 req/s | ~4,800 |

### Tenant Sweep

Fixed: 2 req/s per tenant, 600s duration, seed 42.

| Experiment | Tenants | Aggregate rate | Total requests (approx) |
|---|---|---|---|
| `1T_agg_2rps` | 1 | 2 req/s | ~1,200 |
| `2T_agg_4rps` | 2 | 4 req/s | ~2,400 |
| `3T_agg_6rps` | 3 | 6 req/s | ~3,600 |
| `4T_agg_8rps` | 4 | 8 req/s | ~4,800 |
| `5T_agg_10rps` | 5 | 10 req/s | ~6,000 |

---

## Metrics

### Latency (measured client-side from SSE stream timestamps)

| Metric | Definition |
|---|---|
| TTFT | Time from HTTP request send to first token chunk received |
| ITL | Wall-clock gap between consecutive token chunks (each gap is one sample; flattened across all requests) |
| E2E | Time from HTTP request send to final token chunk received |

### Throughput

| Metric | Definition |
|---|---|
| Output tokens/sec | Total output tokens / wall-clock span |
| Input tokens/sec | Total input tokens / wall-clock span |
| Requests/sec | Completed requests / wall-clock span |

### Residency (patched variant only)

| Metric | Definition |
|---|---|
| `residency_token_seconds` | Cumulative token-seconds of KV-cache occupancy per tenant, scraped from Prometheus `/metrics` endpoint at experiment end |

### Statistics reported

For each latency metric: mean, min, max, median (p50), p90, p99.

---

## Controls for Fair Comparison

| Control | How |
|---|---|
| Same workload | Identical workload spec YAML (same seed, rate, tenants, token counts) |
| Same model weights | Both pods mount the same `vllm-cache-pvc` |
| Same hardware class | Both pods request `nvidia.com/gpu: 1`, scheduled on same node type |
| Simultaneous execution | Both Jobs launched at the same time, run in parallel |
| Same driver code | Both variants use the same `residency-vllm-observe` driver image |
| Same server config | Identical `vllm serve` flags (no extra args on either variant) |
| No contention | Each variant gets a dedicated GPU (no GPU sharing) |

---

## Output Artifacts

Each experiment produces per variant:

| File | Contents |
|---|---|
| `summary.json` | Per-tenant + overall latency/throughput statistics |
| `requests.csv` | Per-request raw data: `tenant_id, request_idx, ttft_ms, mean_itl_ms, p50_itl_ms, p95_itl_ms, p99_itl_ms, e2e_ms, num_output_tokens, start_time` |
| `trace.yaml` | TraceV2 metadata header (blis observe native output) |
| `trace.csv` | Per-request trace with server-reported token counts |
| `trace.itl.csv` | Per-chunk ITL timestamps |

---

## Key Findings

- At non-saturated load (1–4 req/s/tenant with 2 tenants), the residency
  instrumentation adds **~0–1% E2E overhead** (median across all per-tenant
  p50 comparisons).
- Both variants exhibit identical scaling behavior up to the saturation point.
- The overhead is most directly visible in ITL (~0.5–1%) since the residency
  accounting runs in the decode loop and ITL measures per-step duration.
- Adding more tenants at the same aggregate rate does not increase overhead —
  the per-step accounting cost is proportional to the number of active tenants
  but remains negligible.
