# Reproducing the A/B Residency Overhead Experiment

This document describes the full pipeline from patch application through figure
generation.  Following these steps exactly should reproduce the sweep results
in `results/sweep_rate/` and `results/sweep_tenants/`.

---

## Prerequisites

| Requirement | Version / Details |
|---|---|
| OpenShift / Kubernetes cluster | Tested on OpenShift 4.x with `oc` CLI |
| GPU nodes | 2 nodes with NVIDIA GPU (1 GPU per experiment pod) |
| PVCs | `vllm-cache-pvc` (model weights, RWX), `data-pvc` (results, RWX) |
| HuggingFace token | Stored as secret `hf-token` with key `token` |
| Container registry | Push access to `ghcr.io/inference-sim/` (or substitute your own) |
| Local tools | `python3`, `matplotlib`, `numpy`, `oc` or `kubectl` |

---

## 1. The Patch

We patch 3 files in stock vLLM v0.23.0 to add a per-tenant KV-cache residency
Prometheus counter (`vllm:residency_token_seconds_total`).

### Patched files

```
vllm/v1/request.py          — adds tenant_id property (extracted from vllm_xargs)
vllm/v1/engine/core.py      — calls _accumulate_residency() after every scheduler step
vllm/v1/core/kv_cache_manager.py — tracks per-tenant fractional block ownership
```

### What the patch does

After each scheduler step (both prefill and decode), the engine multiplies each
tenant's resident token count by the step's wall-clock duration and increments a
Prometheus counter.  Cross-tenant prefix-cache sharing is handled by fractional
accounting (1/k split across k holders of a shared block).

Full patch specification: `docs/vllm-patch-spec.md`

---

## 2. Building Container Images

### Patched vLLM server

```bash
docker build -t ghcr.io/inference-sim/residency-vllm:0.23.1-residency .
docker push ghcr.io/inference-sim/residency-vllm:0.23.1-residency
```

This overlays the 3 patched Python files onto the stock `vllm/vllm-openai:v0.23.0`
base image (see `Dockerfile`).

### Workload driver client

```bash
docker build -f Dockerfile.client -t ghcr.io/inference-sim/residency-vllm-client:latest .
docker push ghcr.io/inference-sim/residency-vllm-client:latest
```

Contains `workload_driver.py` and its dependency (`aiohttp`).

---

## 3. Cluster Setup

### PVCs

```yaml
# vllm-cache-pvc: stores HuggingFace model weights (persists across experiments)
# data-pvc: stores experiment results (summary.json, requests.csv)
```

Both must be RWX (ReadWriteMany) since patched and vanilla pods run concurrently
and share the same PVCs.

### HuggingFace secret

```bash
oc create secret generic hf-token --from-literal=token=hf_YOUR_TOKEN_HERE
```

### Model pre-download (optional, saves startup time)

The first experiment will auto-download `Qwen/Qwen3-14B` into `vllm-cache-pvc`.
Subsequent experiments reuse the cached weights.

---

## 4. Running Experiments

### Single A/B experiment

```bash
./run_ab_experiment.sh --duration 600 --rate 2.0 --tenants "tenant_A,tenant_B,tenant_C,tenant_D,tenant_E" --seed 42
```

This launches two Kubernetes Jobs simultaneously:
- `residency-experiment-patched` — uses `ghcr.io/inference-sim/residency-vllm:0.23.1-residency`
- `residency-experiment-vanilla` — uses `vllm/vllm-openai:v0.23.0`

Each Job is a single pod with two containers (server + driver) sharing
`localhost` via `shareProcessNamespace: true`.  Both receive identical workload
parameters and random seed for fair comparison.

Results are downloaded to `results/patched/` and `results/vanilla/`.

### Full sweep

```bash
./run_sweep.sh --duration 600 --seed 42
```

Default sweep configuration:

**Rate sweep** (5 tenants fixed, vary per-tenant rate):

| Per-tenant rate | Aggregate rate | Tenants | Output dir |
|---|---|---|---|
| 1 req/s | 5 rps | 5 | `results/sweep_rate/agg_5rps/` |
| 2 req/s | 10 rps | 5 | `results/sweep_rate/agg_10rps/` |
| 3 req/s | 15 rps | 5 | `results/sweep_rate/agg_15rps/` |
| 4 req/s | 20 rps | 5 | `results/sweep_rate/agg_20rps/` |

**Tenant sweep** (2 req/s per tenant fixed, vary tenant count):

| Tenants | Aggregate rate | Per-tenant rate | Output dir |
|---|---|---|---|
| 1 | 2 rps | 2 req/s | `results/sweep_tenants/1T_agg_2rps/` |
| 2 | 4 rps | 2 req/s | `results/sweep_tenants/2T_agg_4rps/` |
| 3 | 6 rps | 2 req/s | `results/sweep_tenants/3T_agg_6rps/` |
| 4 | 8 rps | 2 req/s | `results/sweep_tenants/4T_agg_8rps/` |
| 5 | 10 rps | 2 req/s | `results/sweep_tenants/5T_agg_10rps/` |

Total: 9 experiments × 10 minutes each.

### Custom sweep

```bash
# Different rate points
./run_sweep.sh --rates "1,2,4,6,8" --rate-tenants 5

# Different tenant counts
./run_sweep.sh --tenants "1,3,5,10" --tenant-rate 2.0

# Only one dimension
./run_sweep.sh --only-rate
./run_sweep.sh --only-tenants
```

---

## 5. Experiment Specifications

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen3-14B |
| GPU | 1× NVIDIA GPU per variant (2 total for A/B) |
| Prompt length | 1024 words = exactly 1024 tokens + 5 chat-template overhead = 1029 input tokens |
| Max output tokens | 128 (always hit — output is exactly 128 tokens) |
| Arrival process | Poisson (independent per tenant) |
| Duration | 600 seconds per experiment |
| Random seed | 42 (tenant seeds: 42, 43, 44, ...) |
| Server version (patched) | `ghcr.io/inference-sim/residency-vllm:0.23.1-residency` |
| Server version (vanilla) | `vllm/vllm-openai:v0.23.0` |

### Workload driver details

- Each tenant runs an independent Poisson process with rate R req/s
- Inter-arrival times: `random.expovariate(R)` (exponentially distributed)
- Each tenant gets a deterministic seed (base_seed + tenant_index) for reproducibility
- Requests are streamed (SSE); metrics measured client-side on token arrival timestamps
- TTFT: time from request send to first token chunk received
- ITL: wall-clock gap between consecutive token chunks (flattened across all requests)
- E2E: time from request send to final token chunk received

---

## 6. Output Format

Each experiment produces two files per variant:

```
summary.json  — per-tenant and overall latency/throughput statistics
requests.csv  — per-request raw data (tenant_id, ttft, itl, e2e, timestamps)
```

Directory structure after full sweep:

```
results/
├── sweep_rate/
│   ├── agg_5rps/{patched,vanilla}/summary.json
│   ├── agg_10rps/{patched,vanilla}/summary.json
│   ├── agg_15rps/{patched,vanilla}/summary.json
│   ├── agg_20rps/{patched,vanilla}/summary.json
│   └── figure_rate_sweep.png
└── sweep_tenants/
    ├── 1T_agg_2rps/{patched,vanilla}/summary.json
    ├── 2T_agg_4rps/{patched,vanilla}/summary.json
    ├── 3T_agg_6rps/{patched,vanilla}/summary.json
    ├── 4T_agg_8rps/{patched,vanilla}/summary.json
    ├── 5T_agg_10rps/{patched,vanilla}/summary.json
    └── figure_tenant_sweep.png
```

---

## 7. Generating Figures

After experiments complete:

```bash
python3 generate_figures.py --results-dir ./results
```

This produces:
- `results/sweep_rate/figure_rate_sweep.png` — Latency vs per-tenant rate (log scale)
- `results/sweep_tenants/figure_tenant_sweep.png` — Latency vs tenant count (linear)

Both figures show TTFT, ITL, and E2E (p50) for "vLLM + Residency" vs "Stock vLLM"
with the computed overhead percentage in the caption.

Requirements for figure generation:

```bash
pip install matplotlib numpy
```

---

## 8. Key Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds patched vLLM server image |
| `Dockerfile.client` | Builds workload driver image |
| `vllm/v1/engine/core.py` | Patched engine with residency accumulation |
| `vllm/v1/core/kv_cache_manager.py` | Patched KV cache with per-tenant tracking |
| `vllm/v1/request.py` | Patched request with tenant_id property |
| `workload_driver.py` | Poisson workload generator + metrics collector |
| `k8s/ab-experiment.yaml` | Kubernetes Jobs for both variants |
| `run_ab_experiment.sh` | Runs one A/B experiment |
| `run_sweep.sh` | Orchestrates rate + tenant sweeps |
| `generate_figures.py` | Produces comparison figures from results |
| `docs/vllm-patch-spec.md` | Exact patch instructions for vLLM v0.23.0 |

---

## 9. Troubleshooting

**Model download slow on first run**: The first experiment downloads ~28GB of
model weights.  Subsequent runs reuse `vllm-cache-pvc`.

**JSON download corruption**: The `oc run --rm -i` command appends "pod X deleted"
to stdout.  The scripts filter this with `sed 's/pod ".*" deleted//g'`.

**Credential expiry**: If `oc` session expires mid-sweep, re-authenticate and
re-run `run_sweep.sh`.  It's safe to re-run — the script cleans up before each
experiment.

**GPU scheduling**: Both variants need a GPU simultaneously.  Ensure the cluster
has at least 2 free GPUs before starting an A/B experiment.
