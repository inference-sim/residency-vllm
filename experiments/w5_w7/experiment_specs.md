# Experiment Specifications: Asymmetric Fairness (W5/W7)

## Objective

Validate that the residency counter faithfully measures per-tenant unfairness
under asymmetric workloads. Two asymmetry dimensions are tested:

- **W5 (Rate Asymmetry)**: One tenant sends 5x more requests than the other,
  with identical prompt lengths. Expected disparity ratio rho_mt ~ 5.
- **W7 (Prompt-Length Asymmetry)**: Both tenants send at equal rates, but one
  uses 16x longer prompts. Expected disparity ratio rho_mt >> 1.

A secondary goal is sim-to-real fidelity: replaying real traces through the
BLIS simulator and comparing rho_mt.

---

## Hardware

| Resource | Specification |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (1 per experiment) |
| GPU allocation | `nvidia.com/gpu: 1` per Kubernetes pod |
| Cluster | OpenShift 4.x on bare-metal GPU nodes |
| Storage | Shared PVCs (RWX) for model weights and results |

---

## Server Configuration

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-14B` |
| Precision | BF16 (auto) |
| vLLM version | `ghcr.io/inference-sim/residency-vllm:0.23.0-residency-v2` |
| Server command | `vllm serve --model Qwen/Qwen3-14B --port 8000` |
| max_num_seqs | 256 (default) |
| max_num_batched_tokens | 8192 (auto, chunked prefill enabled) |
| enable_chunked_prefill | True (auto) |
| enable_prefix_caching | True (auto) |
| KV cache capacity | 288,608 tokens (18,038 blocks of 16) |
| block_size | 16 |
| GPU memory utilization | 0.90 (default) |

These values are captured in `server_config.json` at the repo root.

---

## Workloads

### W5: Rate Asymmetry

| Parameter | tenantA | tenantB |
|---|---|---|
| Rate fraction | 1/6 (0.16667) | 5/6 (0.83333) |
| Prompt tokens | 1024 | 1024 |
| Output tokens | 128 | 128 |
| Arrival process | Poisson | Poisson |

Both tenants use identical prompt lengths. The only asymmetry is the arrival
rate (5:1 split favoring tenantB).

### W7: Prompt-Length Asymmetry

| Parameter | tenantA | tenantB |
|---|---|---|
| Rate fraction | 0.5 | 0.5 |
| Prompt tokens | 256 | 4096 |
| Output tokens | 128 | 128 |
| Arrival process | Poisson | Poisson |

Both tenants send at equal rates. The only asymmetry is prompt length (16:1
ratio favoring tenantB's memory consumption).

### Common Parameters

| Parameter | Value |
|---|---|
| Driver | [blis observe](https://github.com/inference-sim/inference-sim) |
| Streaming | Yes (SSE) |
| Duration | 600 seconds per cell |
| Random seed | 42 |
| Warmup | None |
| Workload spec format | WorkloadSpec YAML v2 |

---

## Experiment Matrix

### Rate Sweep (both W5 and W7)

Aggregate rates: 6, 12, 18, 24, 30 req/s.

| Cell | Aggregate rate | Duration | Approx total requests |
|---|---|---|---|
| `agg6` | 6 req/s | 600s | ~3,600 |
| `agg12` | 12 req/s | 600s | ~7,200 |
| `agg18` | 18 req/s | 600s | ~10,800 |
| `agg24` | 24 req/s | 600s | ~14,400 |
| `agg30` | 30 req/s | 600s | ~18,000 |

Total: 10 cells (5 W5 + 5 W7), patched server only (no vanilla baseline needed).

---

## Metrics

### Primary Metric: Disparity Ratio (rho_mt)

```
rho_mt = max(residency per tenant) / min(residency per tenant)
```

Where `residency_token_seconds` for a tenant is the cumulative token-seconds of
KV-cache occupancy (tokens_resident x time_held, integrated over the experiment).

### Latency Metrics

| Metric | Definition |
|---|---|
| TTFT | Time from request send to first token chunk |
| ITL | Wall-clock gap between consecutive token chunks |
| E2E | Time from request send to final token chunk |

### Saturation Knee

The aggregate rate where E2E latency acceleration (second derivative) is maximum.
Detected algorithmically from the E2E median vs rate curve.

---

## Simulator Replay Configuration

After real experiments complete, traces are replayed through the BLIS simulator
for fidelity comparison.

| Sim Parameter | Value | Source |
|---|---|---|
| Binary | `blis-kvtime` (patches 00-03 on inference-sim) |  |
| `-scheduler` | fcfs | Matches vLLM default |
| `-hardware` | H100 | From cluster GPU labels |
| `-tp` | 1 | From vLLM config |
| `-total-kv-blocks` | 18038 | From `server_config.json` (288608 tokens / 16) |
| `-model` | qwen/qwen3-14b | Same model |
| `-model-config-dir` | inference-sim/model_configs/qwen3-14b | HF config.json |
| `-hw-config` | inference-sim/hardware_config.json | H100 FLOPS/bandwidth |
| `-latency-backend` | trained-physics | Fitted latency model |
| `-trace-header` | `<cell>/trace.yaml` | From real experiment |
| `-trace-data` | `<cell>/trace.csv` | From real experiment |

### Hardcoded in blis-kvtime (not configurable via CLI)

| Parameter | Value | vLLM equivalent |
|---|---|---|
| max_running_reqs | 256 | max_num_seqs=256 (matches) |
| max_scheduled_tokens | 32768 | max_num_batched_tokens=8192 (no impact: all prompts < 8192) |
| long_prefill_threshold | 0 (disabled) | chunked prefill at 8192 (no impact: all prompts < 8192) |

---

## Output Artifacts

### Per cell (`results/{w5,w7}_sweep/agg{rate}/`)

| File | Contents |
|---|---|
| `summary.json` | Per-tenant + overall latency/throughput/residency |
| `requests.csv` | Per-request raw data |
| `trace.yaml` | TraceV2 metadata header |
| `trace.csv` | Per-request trace (input for sim replay) |
| `trace.itl.csv` | Per-chunk ITL timestamps |
| `sim.json` | Simulator output (after replay) |

### Per sweep (`results/{w5,w7}_sweep/`)

| File | Contents |
|---|---|
| `fig_rho_mt_vs_rate.png` | rho_mt (real + sim) vs aggregate rate |
| `fig_residency_absolute.png` | Per-tenant absolute residency comparison |

### Repo root

| File | Contents |
|---|---|
| `server_config.json` | vLLM server parameters (captured from logs) |

---

## Results

### W5 (Rate Asymmetry)

| Rate | rho_mt (real) | rho_mt (sim) | Sim error |
|------|--------------|-------------|-----------|
| 6 | 4.781 | 4.776 | 0.1% |
| 12 | 5.105 | 5.098 | 0.2% |
| 18 | 4.935 | 4.943 | 0.2% |
| 24 | 4.958 | 4.979 | 0.4% |
| 30 | 4.964 | 4.977 | 0.2% |

rho_mt is stable around 4.8–5.1 across all rates, consistent with the 5:1
rate split. The slight deviation from exactly 5.0 comes from shared decode
overhead: both tenants pay the same per-token decode cost, which compresses
the ratio slightly at low load (where decode dominates) and amplifies it
slightly at high load (where queuing reflects the rate asymmetry more purely).

### W7 (Prompt-Length Asymmetry)

| Rate | rho_mt (real) | rho_mt (sim) | Sim error |
|------|--------------|-------------|-----------|
| 6 | 13.645 | 12.725 | 6.7% |
| 12 | 12.299 | 12.415 | 0.9% |
| 18 | 12.233 | 12.346 | 0.9% |
| 24 | 12.453 | 12.565 | 0.9% |
| 30 | — | — | (server saturated) |

rho_mt is around 12–14x, well above 1 but below the raw 16:1 prompt-length
ratio. The compression happens because residency = tokens × time, and while
tenantB has 16x more prompt tokens, both tenants generate 128 output tokens
at the same decode rate. The decode phase contributes equally to both tenants'
residency, pulling the ratio below 16.

W7 agg30 failed: 30 req/s with 4096-token prompts (~123K input tokens/s)
exceeds the single H100's sustainable throughput, causing the driver to time
out before results are written.

### Sim Fidelity

- **rho_mt error < 1%** for all W5 cells and W7 at rates >= 12
- W7 agg6 shows 6.7% error — at very low load the sim's latency model handles
  the 4096-token prefills differently (they dominate the schedule at low
  concurrency, and the trained-physics model underestimates prefill time for
  long sequences)
- **Absolute residency diverges** at high load (sim reports 70–77% lower values
  at agg18–30) because the sim's latency model processes requests faster than
  real vLLM under contention. The ratio is preserved because both tenants are
  affected equally.

### Saturation Knee

Detected at agg18 for W5 (E2E latency jumps from ~2s at agg6 to ~14.5s at
agg18–30, indicating the system enters saturation between agg12 and agg18).
