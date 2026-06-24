# Reproducing the Asymmetric Fairness Experiment (W5/W7)

This document describes the full pipeline from experiment execution through
figure generation for the asymmetric fairness workloads.

---

## Prerequisites

| Requirement | Details |
|---|---|
| OpenShift cluster | With `oc` CLI authenticated |
| GPU node | 1x NVIDIA H100 80GB |
| PVCs | `vllm-cache-pvc` (model weights), `data-pvc` (results) |
| HuggingFace token | Secret `hf-token` with key `token` |
| Container images | `ghcr.io/inference-sim/residency-vllm:0.23.0-residency-v2`, `ghcr.io/inference-sim/residency-vllm-observe:0.23.0-residency-v2` |
| Local tools | `python3`, `matplotlib`, `numpy`, `oc` |
| Simulator | `blis-kvtime` binary (for sim replay) |
| inference-sim repo | Clone with `model_configs/` and `hardware_config.json` |

---

## 1. Capture Server Config

Before running experiments, capture the vLLM server parameters from a running
job's logs and save to `server_config.json` at the repo root:

```bash
# Start a quick test job, then:
oc logs job/residency-experiment-patched -c vllm | grep -E "max_num_batched|KV cache size"
```

Expected output:
```
Chunked prefill is enabled with max_num_batched_tokens=8192.
GPU KV cache size: 288,608 tokens
```

Save as `server_config.json`:
```json
{
  "vllm_version": "0.23.0",
  "model": "Qwen/Qwen3-14B",
  "max_seq_len": 40960,
  "tensor_parallel_size": 1,
  "enable_chunked_prefill": true,
  "max_num_batched_tokens": 8192,
  "enable_prefix_caching": true,
  "block_size": 16,
  "kv_cache_tokens": 288608,
  "num_gpu_blocks": 18038
}
```

---

## 2. Run the Experiment

All commands below assume you are in the **repo root** (`residency-vllm/`).

```bash
# Full experiment (both W5 and W7, all 5 rate points each)
experiments/w5_w7/run_fairness_experiment.sh --duration 600 --seed 42

# Or run individually:
experiments/w5_w7/run_fairness_experiment.sh --duration 600 --seed 42 --only-w5
experiments/w5_w7/run_fairness_experiment.sh --duration 600 --seed 42 --only-w7

# Custom rates:
experiments/w5_w7/run_fairness_experiment.sh --rates "6,12,18" --only-w5
```

### What it does

For each rate in {6, 12, 18, 24, 30} req/s:

1. Generates a WorkloadSpec YAML with the appropriate asymmetry (W5 or W7)
2. Uploads the spec to `data-pvc`
3. Launches a single Kubernetes Job (patched vLLM + blis observe driver)
4. Waits for completion
5. Downloads results to `results/{w5,w7}_sweep/agg{rate}/`
6. Prints per-tenant residency and rho_mt

### Output structure

```
results/
├── w5_sweep/
│   ├── agg6/{summary.json, requests.csv, trace.yaml, trace.csv, trace.itl.csv}
│   ├── agg12/...
│   ├── agg18/...
│   ├── agg24/...
│   └── agg30/...
└── w7_sweep/
    ├── agg6/...
    ├── agg12/...
    ├── agg18/...
    ├── agg24/...
    └── agg30/...
```

---

## 3. Sim Replay

After the real experiment completes, replay traces through the BLIS simulator:

```bash
# Both sweeps
experiments/w5_w7/run_sim_replay.sh

# Or individually
experiments/w5_w7/run_sim_replay.sh --only-w5
experiments/w5_w7/run_sim_replay.sh --only-w7
```

### Prerequisites for replay

1. `blis-kvtime` binary built from inference-sim with patches 00-03
   (auto-detected at `~/Downloads/papers/residency/figures/vllmparityexperiment/blis-kvtime`)
2. inference-sim repo with `model_configs/qwen3-14b/config.json` and `hardware_config.json`
   (auto-detected at `../inference-sim` relative to repo root)
3. `server_config.json` at repo root (provides `-total-kv-blocks`)

### What it does

For each cell:
1. Reads `trace.yaml` + `trace.csv` (the real trace)
2. Passes them to `blis-kvtime` with matching server config
3. Writes `sim.json` alongside the real results
4. Prints per-tenant comparison (absolute residency + rho_mt error)

### Building blis-kvtime

```bash
cd /path/to/inference-sim
git apply /path/to/patches/00-tracked.patch
git apply /path/to/patches/01-untracked.patch
git apply /path/to/patches/02-justitia-equinox.patch
git apply /path/to/patches/03-trace-replay-parity.patch
cd cmd/blis-kvtime && go build -o blis-kvtime .
```

---

## 4. Generate Figures

```bash
python3 experiments/w5_w7/generate_fairness_figure.py

# Or specific workload:
python3 experiments/w5_w7/generate_fairness_figure.py --workload w5
python3 experiments/w5_w7/generate_fairness_figure.py --workload w7
```

### Output

| File | Description |
|---|---|
| `results/w5_sweep/fig_rho_mt_vs_rate.png` | W5 disparity ratio (real + sim) vs aggregate rate |
| `results/w5_sweep/fig_residency_absolute.png` | W5 per-tenant absolute residency comparison |
| `results/w7_sweep/fig_rho_mt_vs_rate.png` | W7 disparity ratio (real + sim) vs aggregate rate |
| `results/w7_sweep/fig_residency_absolute.png` | W7 per-tenant absolute residency comparison |

### Requirements

```bash
pip install matplotlib numpy
```

---

## 5. Verification

### Check rho_mt from summary.json

```bash
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
pt = d['per_tenant']
res = {t: v['residency_token_seconds'] for t, v in pt.items()}
rho = max(res.values()) / min(res.values())
for t, r in sorted(res.items()):
    print(f'  {t}: {r:,.0f} token-s')
print(f'  rho_mt = {rho:.3f}')
" results/w5_sweep/agg6/summary.json
```

### Expected rho_mt values

| Sweep | Rate | Expected rho_mt |
|---|---|---|
| W5 | all | ~4.8–5.1 (near the 5:1 rate ratio) |
| W7 | 6–24 | ~12–14 (prompt-length dominated) |

### Sim fidelity check

```bash
python3 -c "
import json
real = json.load(open('results/w5_sweep/agg6/summary.json'))
sim = json.load(open('results/w5_sweep/agg6/sim.json'))
pt = real['per_tenant']
ts = sim['tenants']
rho_real = max(v['residency_token_seconds'] for v in pt.values()) / min(v['residency_token_seconds'] for v in pt.values())
rho_sim = max(v['kv_time_token_us'] for v in ts.values()) / min(v['kv_time_token_us'] for v in ts.values())
print(f'rho_mt real={rho_real:.3f} sim={rho_sim:.3f} error={abs(rho_real-rho_sim)/rho_real*100:.1f}%')
"
```

Expected: < 1% rho_mt error.

---

## 6. Key Files

| File | Purpose |
|---|---|
| `run_fairness_experiment.sh` | Orchestrates W5/W7 rate sweep on the cluster |
| `run_sim_replay.sh` | Replays traces through blis-kvtime simulator |
| `generate_fairness_figure.py` | Produces rho_mt and absolute residency figures |
| `../../server_config.json` | vLLM server params (total KV blocks, etc.) |
| `../../k8s/ab-experiment.yaml` | Base Job YAML (patched half extracted by awk) |

---

## 7. Troubleshooting

**W7 experiments take longer**: The 4096-token prompts in W7 consume more GPU
memory, causing higher queuing at high rates. Allow up to 20 minutes per cell
at agg24/agg30.

**sim.json not produced**: Ensure `blis-kvtime` binary is built with patch 03
(trace-replay-parity). Without it, the `-trace-header`/`-trace-data` flags
don't exist.

**rho_mt < expected for W5**: At very low load, both tenants may have similar
per-request residency (dominated by decode time), compressing rho_mt below the
theoretical 5:1 rate ratio.

**Absolute residency mismatch (sim vs real)**: Expected. The sim's latency model
processes requests faster than real vLLM under load, so absolute token-seconds
are lower. The ratio (rho_mt) is preserved because both tenants are affected
equally.

**server_config.json missing**: Capture from a running job's vLLM logs (see
step 1). The key values are `kv_cache_tokens` and `block_size`.
