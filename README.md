# residency-vllm

Per-tenant KV-cache residency instrumentation for vLLM. Exposes a Prometheus
counter (`vllm:residency_token_seconds_total`) that accumulates token-seconds
of GPU KV-cache occupancy per tenant, with fair splitting of shared prefix-cache
blocks.

**Released image:** `ghcr.io/inference-sim/residency-vllm:0.23.1-residency`
(based on `vllm/vllm-openai:v0.23.0`)

Includes an A/B experiment framework that runs identical Poisson workloads
against the patched server and stock vLLM simultaneously, demonstrating ~1%
E2E latency overhead from the instrumentation.

## How it works

Three patched Python files are overlaid onto the official `vllm/vllm-openai:v0.23.0`
Docker image via `cp -r`. No compilation required — the base image ships all
CUDA/C++ extensions pre-built.

```
vllm/v1/
├── request.py              ← extracts tenant_id from vllm_xargs
├── engine/core.py          ← step timing + _accumulate_residency()
└── core/kv_cache_manager.py ← residency_holders + tenant_resident_tokens
```

After each scheduler step, the engine multiplies each tenant's resident KV-cache
token count by the step duration and increments a Prometheus counter.  Shared
prefix-cache blocks are split fractionally (1/k per holder).

## Prerequisites

- OpenShift / Kubernetes cluster with GPU nodes (`nvidia.com/gpu`)
- `oc` CLI configured and logged in
- HuggingFace token with access to `Qwen/Qwen3-14B`
- 2 available GPUs (for A/B experiments — one per variant)
- PVCs: `vllm-cache-pvc` (model weights), `data-pvc` (results)

## Quick start

### 1. Create secrets and PVCs

```bash
oc create secret generic hf-token --from-literal=token=<YOUR_HF_TOKEN>
```

### 2. Build and push images

```bash
# Patched vLLM server
docker build -t ghcr.io/inference-sim/residency-vllm:0.23.1-residency .
docker push ghcr.io/inference-sim/residency-vllm:0.23.1-residency

# Workload driver
docker build -f Dockerfile.client -t ghcr.io/inference-sim/residency-vllm-client:latest .
docker push ghcr.io/inference-sim/residency-vllm-client:latest
```

### 3. Run a single A/B experiment

```bash
./run_ab_experiment.sh --duration 600 --rate 2.0 --tenants "tenant_A,tenant_B,tenant_C,tenant_D,tenant_E"
```

Results are downloaded to `results/patched/` and `results/vanilla/`.

### 4. Run the full sweep

```bash
./run_sweep.sh --duration 600
```

Runs 9 experiments (4 rate points + 5 tenant points) and saves results to
`results/sweep_rate/` and `results/sweep_tenants/`.

### 5. Generate figures

```bash
pip install matplotlib numpy
python3 generate_figures.py
```

Outputs:
- `results/sweep_rate/figure_rate_sweep.png`
- `results/sweep_tenants/figure_tenant_sweep.png`

## A/B Experiment Design

Two Kubernetes Jobs run simultaneously on separate GPUs:

| Variant | Server Image | Difference |
|---|---|---|
| Patched | `ghcr.io/inference-sim/residency-vllm:0.23.1-residency` | Residency counter enabled |
| Vanilla | `vllm/vllm-openai:v0.23.0` | Stock vLLM, no instrumentation |

Both receive identical workload (same seed, rate, tenants, prompt length) via
a co-located driver container.  Metrics are measured client-side from SSE
streaming token timestamps.

### Default experiment parameters

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen3-14B |
| Prompt length | 1024 tokens |
| Max output tokens | 128 |
| Arrival process | Poisson (per-tenant) |
| Duration | 600s |
| Seed | 42 |

### Sweep configurations

**Rate sweep** — 5 tenants fixed, per-tenant rate varies (1, 2, 3, 4 req/s):

```
results/sweep_rate/agg_5rps/    (1 req/s × 5 tenants)
results/sweep_rate/agg_10rps/   (2 req/s × 5 tenants)
results/sweep_rate/agg_15rps/   (3 req/s × 5 tenants)
results/sweep_rate/agg_20rps/   (4 req/s × 5 tenants)
```

**Tenant sweep** — 2 req/s per tenant fixed, tenant count varies (1–5):

```
results/sweep_tenants/1T_agg_2rps/
results/sweep_tenants/2T_agg_4rps/
results/sweep_tenants/3T_agg_6rps/
results/sweep_tenants/4T_agg_8rps/
results/sweep_tenants/5T_agg_10rps/
```

## Repository layout

```
.
├── Dockerfile                    # Patched vLLM server image
├── Dockerfile.client             # Workload driver image
├── requirements-client.txt       # Driver dependencies (aiohttp)
├── workload_driver.py            # Poisson workload generator + metrics
├── run_ab_experiment.sh          # Single A/B experiment
├── run_sweep.sh                  # Rate + tenant sweep orchestration
├── generate_figures.py           # Produce comparison figures
├── reproduce.md                  # Full reproduction instructions
├── k8s/
│   ├── ab-experiment.yaml        # Two Jobs (patched + vanilla)
│   └── deployment.yaml           # Standalone deployment + service
├── vllm/v1/                      # Patched files (overlay source)
│   ├── request.py
│   ├── engine/core.py
│   └── core/kv_cache_manager.py
├── docs/
│   ├── vllm-residency-design.md  # Architecture + invariants
│   ├── vllm-patch-spec.md        # Exact patch instructions
│   └── deployment-plan.md        # Deployment walkthrough
└── results/                      # Experiment outputs (not committed)
    ├── sweep_rate/
    └── sweep_tenants/
```

## Metrics collected

### Per-request (requests.csv)

`tenant_id, request_idx, ttft_ms, tpot_ms, mean_itl_ms, e2e_ms, num_input_tokens, num_output_tokens, start_time`

### Per-tenant and overall (summary.json)

- **TTFT** — time to first token (prefill + queue wait)
- **ITL** — inter-token latency (flattened across all requests)
- **TPOT** — time per output token: (last_token - first_token) / (n - 1)
- **E2E** — end-to-end latency
- **Throughput** — input/output/total tokens per second, requests per second
- **Residency** — `residency_token_seconds` scraped from Prometheus endpoint

Statistics reported: mean, min, max, median, p90, p99.

## Configuration

| Environment variable | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace access token (via k8s secret) |
| `HF_HOME` | Model cache directory (defaults to `/cache/huggingface`) |
| `PROMETHEUS_MULTIPROC_DIR` | Required for patched variant (multiprocess prometheus) |

## Useful PromQL queries

| Query | Meaning |
|---|---|
| `vllm:residency_token_seconds_total` | Raw cumulative per tenant |
| `rate(vllm:residency_token_seconds_total[1m])` | Avg resident tokens (instantaneous) |
| `rate(...{tenant_id="A"}[5m]) / sum(rate(...[5m]))` | Tenant A's share of total |

## Documentation

- [Experiment specifications](experiment_specs.md) — hardware, model, workload, metrics, and findings
- [Reproduction guide](reproduce.md) — step-by-step pipeline reproduction
- [Design document](docs/vllm-residency-design.md) — architecture, incremental tracking, conservation invariants
- [Patch specification](docs/vllm-patch-spec.md) — exact code changes with line numbers
- [Deployment plan](docs/deployment-plan.md) — end-to-end deployment walkthrough
