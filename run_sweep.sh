#!/bin/bash
# run_sweep.sh — Run rate sweep and tenant sweep experiments
#
# Usage:
#   ./run_sweep.sh [OPTIONS]
#
# Options:
#   --duration SEC       Duration per experiment (default: 600)
#   --seed N             Random seed (default: 42)
#   --rates "1,2,3,4"   Per-tenant rates for rate sweep (default: "1,2,3,4")
#   --rate-tenants N     Number of tenants for rate sweep (default: 5)
#   --tenant-rate R      Per-tenant rate for tenant sweep (default: 2.0)
#   --tenants "1,3,4,5" Tenant counts for tenant sweep (default: "1,2,3,4,5")
#   --only-rate          Only run the rate sweep
#   --only-tenants       Only run the tenant sweep
#   --no-wait            Don't wait; just launch (not recommended for sweeps)
#
# Example:
#   ./run_sweep.sh --duration 600 --rates "1,2,3,4" --tenants "1,2,3,4,5"

set -euo pipefail

# --- Defaults ---
DURATION=600
SEED=42
RATES="1,2,3,4"
RATE_TENANTS=5
TENANT_RATE="2.0"
TENANT_COUNTS="1,2,3,4,5"
ONLY_RATE=false
ONLY_TENANTS=false

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --duration)      DURATION="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --rates)         RATES="$2"; shift 2 ;;
    --rate-tenants)  RATE_TENANTS="$2"; shift 2 ;;
    --tenant-rate)   TENANT_RATE="$2"; shift 2 ;;
    --tenants)       TENANT_COUNTS="$2"; shift 2 ;;
    --only-rate)     ONLY_RATE=true; shift ;;
    --only-tenants)  ONLY_TENANTS=true; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# --- Helper: build tenant list for N tenants ---
make_tenant_list() {
  local n=$1
  local names=("tenant_A" "tenant_B" "tenant_C" "tenant_D" "tenant_E"
               "tenant_F" "tenant_G" "tenant_H" "tenant_I" "tenant_J")
  local out=""
  for ((i=0; i<n; i++)); do
    [[ -n "$out" ]] && out+=","
    out+="${names[$i]}"
  done
  echo "$out"
}

# --- Helper: run one A/B experiment and save to a directory ---
run_one() {
  local rate="$1"
  local tenants="$2"
  local dest_dir="$3"

  echo "--- Running: $(echo "$tenants" | tr ',' '\n' | wc -l | tr -d ' ') tenants @ ${rate} req/s (aggregate: $(echo "$tenants" | tr ',' '\n' | wc -l | tr -d ' ' | xargs -I{} python3 -c "print(int({}) * $rate)") req/s) ---"

  "$SCRIPT_DIR/run_ab_experiment.sh" \
    --duration "$DURATION" \
    --rate "$rate" \
    --seed "$SEED" \
    --tenants "$tenants"

  # Copy results to destination
  mkdir -p "$dest_dir/patched" "$dest_dir/vanilla"
  cp "$RESULTS_DIR/patched/summary.json" "$dest_dir/patched/"
  cp "$RESULTS_DIR/patched/requests.csv" "$dest_dir/patched/"
  cp "$RESULTS_DIR/vanilla/summary.json" "$dest_dir/vanilla/"
  cp "$RESULTS_DIR/vanilla/requests.csv" "$dest_dir/vanilla/"

  # Clean up temporary download dirs
  rm -rf "$RESULTS_DIR/patched" "$RESULTS_DIR/vanilla"

  echo "  Saved to $(basename "$(dirname "$dest_dir")")/$(basename "$dest_dir")/"
  echo ""
}

# --- Count experiments ---
EXPERIMENT_NUM=0
TOTAL_EXPERIMENTS=0

if [[ "$ONLY_TENANTS" != "true" ]]; then
  TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + $(echo "$RATES" | tr ',' '\n' | wc -l | tr -d ' ')))
fi
if [[ "$ONLY_RATE" != "true" ]]; then
  TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + $(echo "$TENANT_COUNTS" | tr ',' '\n' | wc -l | tr -d ' ')))
fi

echo "============================================================"
echo "  RESIDENCY A/B SWEEP"
echo "  Duration per experiment: ${DURATION}s"
echo "  Total experiments: ${TOTAL_EXPERIMENTS}"
echo "============================================================"
echo ""

# ==========================================================
# RATE SWEEP: fixed number of tenants, vary per-tenant rate
# ==========================================================
if [[ "$ONLY_TENANTS" != "true" ]]; then
  echo "=== RATE SWEEP (${RATE_TENANTS} tenants, vary rate) ==="
  echo ""

  RATE_TENANT_LIST=$(make_tenant_list "$RATE_TENANTS")

  IFS=',' read -ra RATE_ARRAY <<< "$RATES"
  for RATE in "${RATE_ARRAY[@]}"; do
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
    AGG=$(python3 -c "print(int(${RATE_TENANTS} * $RATE))")
    DEST="$RESULTS_DIR/sweep_rate/agg_${AGG}rps"

    run_one "$RATE" "$RATE_TENANT_LIST" "$DEST"

    echo "Cleanup after experiment ${EXPERIMENT_NUM}"
  done
fi

# ==========================================================
# TENANT SWEEP: fixed per-tenant rate, vary number of tenants
# ==========================================================
if [[ "$ONLY_RATE" != "true" ]]; then
  echo "=== TENANT SWEEP (${TENANT_RATE} req/s per tenant, vary tenants) ==="
  echo ""

  IFS=',' read -ra TENANT_ARRAY <<< "$TENANT_COUNTS"
  for N in "${TENANT_ARRAY[@]}"; do
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
    TENANT_LIST=$(make_tenant_list "$N")
    AGG=$(python3 -c "print(int($N * $TENANT_RATE))")
    DEST="$RESULTS_DIR/sweep_tenants/${N}T_agg_${AGG}rps"

    run_one "$TENANT_RATE" "$TENANT_LIST" "$DEST"

    echo "Cleanup after experiment ${EXPERIMENT_NUM}"
  done
fi

echo "All sweeps done, final cleanup"
echo ""
echo "=== ALL SWEEPS COMPLETE ==="
echo ""
echo "Results saved under:"
if [[ "$ONLY_TENANTS" != "true" ]]; then
  echo "  $RESULTS_DIR/sweep_rate/"
  ls -d "$RESULTS_DIR/sweep_rate"/agg_* 2>/dev/null | sed 's/^/    /'
fi
if [[ "$ONLY_RATE" != "true" ]]; then
  echo "  $RESULTS_DIR/sweep_tenants/"
  ls -d "$RESULTS_DIR/sweep_tenants"/*T_* 2>/dev/null | sed 's/^/    /'
fi
