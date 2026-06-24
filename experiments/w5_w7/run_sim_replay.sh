#!/bin/bash
# run_sim_replay.sh — Replay real traces through the patched simulator (blis-kvtime)
# and compare per-tenant residency (rho_mt) between real and sim.
#
# Usage: ./run_sim_replay.sh [OPTIONS]
#
# Options:
#   --blis-kvtime PATH     Path to blis-kvtime binary (default: auto-detect)
#   --blis-repo PATH       Path to inference-sim clone (for model/hw configs)
#   --results-dir PATH     Path to results directory (default: ./results)
#   --workload w5|w7|both  Which sweep(s) to replay (default: both)
#   --only-w5              Shorthand for --workload w5
#   --only-w7              Shorthand for --workload w7

set -euo pipefail

# --- Defaults ---
BLIS_KVTIME=""
BLIS_REPO=""
RESULTS_DIR=""
WORKLOAD="both"

# --- Arg parsing ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --blis-kvtime) BLIS_KVTIME="$2"; shift 2 ;;
    --blis-repo) BLIS_REPO="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --only-w5) WORKLOAD="w5"; shift ;;
    --only-w7) WORKLOAD="w7"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Auto-detect paths ---
if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="$REPO_ROOT/results"
fi

if [[ -z "$BLIS_REPO" ]]; then
  for candidate in \
    "$REPO_ROOT/../inference-sim" \
    "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim"; do
    if [[ -f "$candidate/hardware_config.json" ]]; then
      BLIS_REPO="$candidate"
      break
    fi
  done
  if [[ -z "$BLIS_REPO" ]]; then
    echo "ERROR: Cannot find inference-sim repo. Set --blis-repo."
    exit 1
  fi
fi

if [[ -z "$BLIS_KVTIME" ]]; then
  for candidate in \
    "/Users/dipanwitaguhathakurta/Downloads/papers/residency/figures/vllmparityexperiment/blis-kvtime" \
    "$SCRIPT_DIR/blis-kvtime"; do
    if [[ -x "$candidate" ]]; then
      BLIS_KVTIME="$candidate"
      break
    fi
  done
  if [[ -z "$BLIS_KVTIME" ]]; then
    echo "ERROR: Cannot find blis-kvtime binary. Set --blis-kvtime."
    exit 1
  fi
fi

MODEL_CONFIG_DIR="$BLIS_REPO/model_configs/qwen3-14b"
HW_CONFIG="$BLIS_REPO/hardware_config.json"

# Validate paths
for path in "$BLIS_KVTIME" "$MODEL_CONFIG_DIR/config.json" "$HW_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Required file not found: $path"
    exit 1
  fi
done

echo "=== Sim Replay: blis-kvtime trace replay ==="
echo "  Binary:       $BLIS_KVTIME"
echo "  BLIS repo:    $BLIS_REPO"
echo "  Results dir:  $RESULTS_DIR"
echo "  Workload:     $WORKLOAD"
echo ""

# --- Load vLLM server config ---
# blis-kvtime's main.go hardcodes max_running_reqs=256 and max_scheduled_tokens=32768
# (chunked prefill disabled). These match vLLM behavior for our prompt sizes
# (W5=1024, W7=4096 — both below vLLM's 8192 chunked-prefill threshold).
# The only param we need to align is total-kv-blocks (KV cache capacity).
SERVER_CONFIG="$REPO_ROOT/server_config.json"
if [[ ! -f "$SERVER_CONFIG" ]]; then
  echo "ERROR: $SERVER_CONFIG not found."
  echo "  Capture it from a running experiment with:"
  echo "    oc logs job/residency-experiment-patched -c vllm | grep -E 'max_num_batched|KV cache size'"
  exit 1
fi

eval "$(python3 - "$SERVER_CONFIG" <<'PYEOF'
import json, sys
config = json.load(open(sys.argv[1]))
num_blocks = config.get('num_gpu_blocks')
if num_blocks is None:
    kv_tokens = config.get('kv_cache_tokens')
    block_size = config.get('block_size', 16)
    if kv_tokens:
        num_blocks = kv_tokens // block_size
if num_blocks:
    print(f'SIM_TOTAL_KV_BLOCKS={num_blocks}')
PYEOF
)"

echo "  Server config: $SERVER_CONFIG"
echo "    total-kv-blocks=${SIM_TOTAL_KV_BLOCKS:-<default>}"
echo ""

# --- replay_one: run blis-kvtime on a single cell ---
# Usage: replay_one <cell_dir>
replay_one() {
  local cell_dir=$1

  local trace_header="$cell_dir/trace.yaml"
  local trace_data="$cell_dir/trace.csv"
  local output="$cell_dir/sim.json"

  # Skip if traces don't exist
  if [[ ! -f "$trace_header" ]] || [[ ! -f "$trace_data" ]]; then
    echo "  SKIP (no trace files): $cell_dir"
    return 0
  fi

  # Build sim flags from server config
  local extra_flags=()
  [[ -n "${SIM_TOTAL_KV_BLOCKS:-}" ]] && extra_flags+=("-total-kv-blocks=$SIM_TOTAL_KV_BLOCKS")

  echo "  Replaying: $cell_dir"
  "$BLIS_KVTIME" \
    -scheduler=fcfs \
    -trace-header="$trace_header" \
    -trace-data="$trace_data" \
    -model=qwen/qwen3-14b \
    -hardware=H100 \
    -tp=1 \
    -model-config-dir="$MODEL_CONFIG_DIR" \
    -hw-config="$HW_CONFIG" \
    -latency-backend=trained-physics \
    -seed=42 \
    -warmup=0 \
    "${extra_flags[@]}" \
    -output="$output"

  if [[ ! -f "$output" ]]; then
    echo "    ERROR: sim.json not produced"
    return 1
  fi

  echo "    Wrote: $output"
  return 0
}

# --- compare_one: print rho_mt comparison for a single cell ---
# Usage: compare_one <cell_dir>
compare_one() {
  local cell_dir=$1
  local summary="$cell_dir/summary.json"
  local sim_output="$cell_dir/sim.json"

  if [[ ! -f "$summary" ]] || [[ ! -f "$sim_output" ]]; then
    return 0
  fi

  python3 - "$cell_dir" "$summary" "$sim_output" <<'PYEOF' || echo "    (comparison failed)"
import json, sys

cell = sys.argv[1]
real = json.load(open(sys.argv[2]))
sim = json.load(open(sys.argv[3]))

# Real side: per_tenant.<tenant>.residency_token_seconds
per_tenant_real = real.get('per_tenant', {})
if not per_tenant_real:
    print('    (no per_tenant data in summary.json)')
    sys.exit(0)

real_residencies = {t: v.get('residency_token_seconds', 0)
                    for t, v in per_tenant_real.items()}

# Sim side: tenants.<tenant>.kv_time_token_us (convert to seconds)
tenants_sim = sim.get('tenants', sim.get('per_tenant', {}))
if not tenants_sim:
    print('    (no tenant data in sim.json)')
    sys.exit(0)

sim_residencies = {}
for t, v in tenants_sim.items():
    if isinstance(v, dict):
        if 'kv_time_token_us' in v:
            sim_residencies[t] = v['kv_time_token_us'] / 1e6
        elif 'residency_token_seconds' in v:
            sim_residencies[t] = v['residency_token_seconds']
        else:
            sim_residencies[t] = 0.0

# Compute rho_mt for both
def rho_mt(residencies):
    vals = [v for v in residencies.values() if v > 0]
    if len(vals) < 2:
        return float('nan')
    return max(vals) / min(vals)

rho_real = rho_mt(real_residencies)
rho_sim = rho_mt(sim_residencies)

# Per-tenant comparison
print(f'    {"Tenant":<10} {"Real (s)":<14} {"Sim (s)":<14} {"Error%":<10}')
print(f'    {"─"*10} {"─"*14} {"─"*14} {"─"*10}')
for t in sorted(set(list(real_residencies.keys()) + list(sim_residencies.keys()))):
    r = real_residencies.get(t, 0)
    s = sim_residencies.get(t, 0)
    err = abs(r - s) / max(r, 1e-9) * 100
    print(f'    {t:<10} {r:<14.2f} {s:<14.2f} {err:<10.1f}')

# rho_mt comparison
rho_err = abs(rho_real - rho_sim) / max(rho_real, 1e-9) * 100
print(f'    rho_mt:    real={rho_real:.3f}  sim={rho_sim:.3f}  error={rho_err:.1f}%')
PYEOF
}

# --- Main loop ---
CELLS_RUN=0
CELLS_SKIPPED=0

run_sweep() {
  local sweep_dir=$1
  local label=$2

  if [[ ! -d "$sweep_dir" ]]; then
    echo "  Directory not found: $sweep_dir (skipping)"
    echo ""
    return
  fi

  echo "========================================="
  echo "  $label"
  echo "========================================="
  echo ""

  for cell_dir in "$sweep_dir"/agg*/; do
    [[ -d "$cell_dir" ]] || continue
    local rate_label
    rate_label=$(basename "$cell_dir")

    echo "--- $label | $rate_label ---"
    if replay_one "$cell_dir"; then
      compare_one "$cell_dir"
      ((CELLS_RUN++)) || true
    else
      ((CELLS_SKIPPED++)) || true
    fi
    echo ""
  done
}

if [[ "$WORKLOAD" == "both" || "$WORKLOAD" == "w5" ]]; then
  run_sweep "$RESULTS_DIR/w5_sweep" "W5: Rate Asymmetry (sim replay)"
fi

if [[ "$WORKLOAD" == "both" || "$WORKLOAD" == "w7" ]]; then
  run_sweep "$RESULTS_DIR/w7_sweep" "W7: Prompt-Length Asymmetry (sim replay)"
fi

# --- Final summary table ---
echo ""
echo "=== Replay Complete ==="
echo "  Cells replayed: $CELLS_RUN"
echo "  Cells skipped:  $CELLS_SKIPPED"
echo ""

# Print aggregate comparison table
if command -v python3 &>/dev/null; then
  echo "=== rho_mt Summary (Real vs Sim) ==="
  echo ""
  python3 - "$RESULTS_DIR" "$WORKLOAD" <<'PYEOF' || true
import json, os, sys

results_dir = sys.argv[1]
workload = sys.argv[2]

print(f'  {"Wkld":<6} {"Rate":<6} {"rho_real":<10} {"rho_sim":<10} {"Error%":<10}')
print(f'  {"─"*6} {"─"*6} {"─"*10} {"─"*10} {"─"*10}')

for sweep in ['w5_sweep', 'w7_sweep']:
    if workload != 'both' and sweep.split('_')[0] != workload:
        continue
    sweep_dir = os.path.join(results_dir, sweep)
    if not os.path.isdir(sweep_dir):
        continue
    label = sweep.split('_')[0].upper()
    for cell in sorted(os.listdir(sweep_dir)):
        cell_dir = os.path.join(sweep_dir, cell)
        summary = os.path.join(cell_dir, 'summary.json')
        sim_out = os.path.join(cell_dir, 'sim.json')
        if not os.path.isfile(summary) or not os.path.isfile(sim_out):
            continue
        try:
            real = json.load(open(summary))
            sim = json.load(open(sim_out))
            # Real rho_mt
            pt = real.get('per_tenant', {})
            real_res = {t: v.get('residency_token_seconds', 0) for t, v in pt.items()}
            vals_r = [v for v in real_res.values() if v > 0]
            rho_r = max(vals_r) / min(vals_r) if len(vals_r) >= 2 else float('nan')
            # Sim rho_mt
            ts = sim.get('tenants', sim.get('per_tenant', {}))
            sim_res = {}
            for t, v in ts.items():
                if isinstance(v, dict):
                    if 'kv_time_token_us' in v:
                        sim_res[t] = v['kv_time_token_us'] / 1e6
                    elif 'residency_token_seconds' in v:
                        sim_res[t] = v['residency_token_seconds']
            vals_s = [v for v in sim_res.values() if v > 0]
            rho_s = max(vals_s) / min(vals_s) if len(vals_s) >= 2 else float('nan')
            err = abs(rho_r - rho_s) / max(rho_r, 1e-9) * 100
            rate = cell.replace('agg', '')
            print(f'  {label:<6} {rate:<6} {rho_r:<10.3f} {rho_s:<10.3f} {err:<10.1f}')
        except Exception:
            pass
PYEOF
fi
echo ""
echo "Done."
