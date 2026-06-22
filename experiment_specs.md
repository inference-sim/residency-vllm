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
| vLLM version (patched) | `ghcr.io/inference-sim/residency-vllm:0.23.1-residency` |
| Server flags | `vllm serve --model Qwen/Qwen3-14B --port 8000` |

---

## Workload

| Parameter | Value |
|---|---|
| Arrival process | Poisson (independent per tenant) |
| Inter-arrival time | `random.expovariate(rate)` per tenant |
| Prompt length | ~1024 tokens (1024 random common English words; actual token count varies slightly by tokenizer) |
| Max output tokens | 128 |
| Streaming | Yes (SSE) |
| Duration | 600 seconds per experiment |
| Random seed | 42 (tenant seeds: 42 + tenant_index) |
| Warmup | None (cold start included) |

### Tenant ID injection

Each request includes a `vllm_xargs` field:

```json
{"vllm_xargs": {"tenant_id": "tenant_A"}}
```

The patched server extracts this for residency accounting. The vanilla server
ignores unknown fields.

---

## Experiment Matrix

### Rate Sweep

Fixed: 5 tenants (`tenant_A` through `tenant_E`), 600s duration, seed 42.

| Experiment | Per-tenant rate | Aggregate rate | Total requests (approx) |
|---|---|---|---|
| `agg_5rps` | 1 req/s | 5 req/s | ~3,000 |
| `agg_10rps` | 2 req/s | 10 req/s | ~6,000 |
| `agg_15rps` | 3 req/s | 15 req/s | ~9,000 |
| `agg_20rps` | 4 req/s | 20 req/s | ~12,000 |

### Tenant Sweep

Fixed: 2 req/s per tenant, 600s duration, seed 42.

| Experiment | Tenants | Aggregate rate | Total requests (approx) |
|---|---|---|---|
| `1T_agg_2rps` | 1 | 2 req/s | ~1,200 |
| `2T_agg_4rps` | 2 | 4 req/s | ~2,400 |
| `3T_agg_6rps` | 3 | 6 req/s | ~3,500 |
| `4T_agg_8rps` | 4 | 8 req/s | ~4,700 |
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
| Same workload | Identical random seed, rate, tenants, prompt length, max tokens |
| Same model weights | Both pods mount the same `vllm-cache-pvc` |
| Same hardware class | Both pods request `nvidia.com/gpu: 1`, scheduled on same node type |
| Simultaneous execution | Both Jobs launched at the same time, run in parallel |
| Same driver code | Both variants use the same `residency-vllm-client` image |
| Same server config | Identical `vllm serve` flags (no extra args on either variant) |
| No contention | Each variant gets a dedicated GPU (no GPU sharing) |

---

## Output Artifacts

Each experiment produces per variant:

| File | Contents |
|---|---|
| `summary.json` | Per-tenant + overall latency/throughput statistics |
| `requests.csv` | Per-request raw data: `tenant_id, request_idx, ttft_ms, tpot_ms, mean_itl_ms, e2e_ms, num_input_tokens, num_output_tokens, start_time` |

---

## Key Findings

- At non-saturated load (1-2 req/s/tenant), the residency instrumentation adds
  **~1% E2E overhead** (median across all per-tenant p50 comparisons).
- Both variants exhibit identical scaling behavior up to the saturation point
  (~3 req/s/tenant for this model on A100).
- Beyond saturation, queueing dominates and small per-step differences are
  amplified (TTFT grows to tens of seconds as requests queue).
- The overhead is most directly visible in ITL (~0.25-2%) since the residency
  accounting runs in the decode loop and ITL measures per-step duration.
