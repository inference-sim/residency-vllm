#!/bin/bash
# run_fairness_experiment.sh — Asymmetric fairness workload experiments
# Runs W5 (rate asymmetry) and W7 (prompt-length asymmetry) workloads across
# a rate sweep on the patched server only. Collects trace files and residency
# data to measure per-tenant unfairness (rho_mt).
#
# Usage: ./run_fairness_experiment.sh [OPTIONS]
#
# Options:
#   --duration SEC          Duration per experiment (default: 600)
#   --seed N                Random seed (default: 42)
#   --rates "6,12,18,24,30" Aggregate rates to sweep (default: "6,12,18,24,30")
#   --workload w5|w7|both   Which workload(s) (default: both)
#   --only-w5               Shorthand for --workload w5
#   --only-w7               Shorthand for --workload w7

set -euo pipefail

# --- Defaults ---
DURATION=600
SEED=42
RATES="6,12,18,24,30"
WORKLOAD="both"

# --- Arg parsing ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --duration) DURATION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --rates) RATES="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --only-w5) WORKLOAD="w5"; shift ;;
    --only-w7) WORKLOAD="w7"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_YAML="$REPO_ROOT/k8s/ab-experiment.yaml"
LOCAL_RESULTS="$REPO_ROOT/results"

# --- Workload spec generators ---

gen_w5_spec() {
  # W5: rate asymmetry — tenantA gets 1/6 of load, tenantB gets 5/6, same prompts (1024)
  local agg_rate=$1 duration=$2 seed=$3
  local horizon
  horizon=$(echo "$duration * 1000000" | bc)
  cat <<EOF
version: "2"
seed: ${seed}
aggregate_rate: ${agg_rate}
horizon: ${horizon}

clients:
  - id: "tenantA"
    tenant_id: "tenantA"
    rate_fraction: 0.16667
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 1024
    output_distribution:
      type: constant
      params:
        value: 128
  - id: "tenantB"
    tenant_id: "tenantB"
    rate_fraction: 0.83333
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 1024
    output_distribution:
      type: constant
      params:
        value: 128
EOF
}

gen_w7_spec() {
  # W7: prompt-length asymmetry — equal rates, tenantA 256 tokens, tenantB 4096
  local agg_rate=$1 duration=$2 seed=$3
  local horizon
  horizon=$(echo "$duration * 1000000" | bc)
  cat <<EOF
version: "2"
seed: ${seed}
aggregate_rate: ${agg_rate}
horizon: ${horizon}

clients:
  - id: "tenantA"
    tenant_id: "tenantA"
    rate_fraction: 0.5
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 256
    output_distribution:
      type: constant
      params:
        value: 128
  - id: "tenantB"
    tenant_id: "tenantB"
    rate_fraction: 0.5
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 4096
    output_distribution:
      type: constant
      params:
        value: 128
EOF
}

# --- run_one: execute a single experiment cell ---
# Usage: run_one <workload_spec_file> <dest_dir> <rate> <duration> <seed>
run_one() {
  local spec_file=$1 dest_dir=$2 rate=$3 duration=$4 seed=$5

  # a. Clean stale data on PVC
  echo "  Clearing /data/residency/patched on PVC..."
  oc run fairness-setup --rm -i --restart=Never --image=busybox \
    --overrides='{
      "spec":{
        "containers":[{
          "name":"fairness-setup",
          "image":"busybox",
          "command":["sh","-c","mkdir -p /data/residency && chmod 777 /data/residency && rm -rf /data/residency/patched && cat > /data/residency/workload.yaml"],
          "stdin": true,
          "securityContext":{"runAsUser":0},
          "volumeMounts":[{"name":"data","mountPath":"/data"}]
        }],
        "volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc"}}]
      }
    }' < "$spec_file" 2>/dev/null || true

  # b. Launch patched-only job (extract first YAML document from ab-experiment.yaml)
  #    The file has --- on line 4 (doc start) and line 127 (separator for vanilla).
  #    awk '/^---$/{c++; next} c==1' extracts only the patched job between them.
  echo "  Launching patched experiment job (rate=${rate}, duration=${duration})..."
  awk '/^---$/{c++; next} c==1' "$JOB_YAML" | \
    sed \
      -e "s|--rate 2.0|--rate ${rate}|g" \
      -e "s|--duration 300|--duration ${duration}|g" \
      -e "s|--seed 42|--seed ${seed}|g" \
    | oc apply -f -

  # c. Wait for pod readiness, then tail logs
  echo "  Waiting for experiment to complete..."
  oc wait --for=condition=ready pod -l variant=patched,experiment=ab-residency --timeout=600s 2>/dev/null || true
  oc logs -f job/residency-experiment-patched -c driver 2>/dev/null || true

  # d. Wait for job completion
  oc wait --for=condition=complete job/residency-experiment-patched --timeout=1800s 2>/dev/null || true

  # e. Download results
  echo "  Downloading results to ${dest_dir}..."
  mkdir -p "$dest_dir"
  local result_files="summary.json requests.csv trace.yaml trace.csv trace.itl.csv"

  for file in $result_files; do
    local pod_suffix
    pod_suffix=$(echo "${file}" | tr '.' '-')
    oc run "fetch-patched-${pod_suffix}" --rm -i --restart=Never --image=busybox \
      --overrides="{
        \"spec\":{
          \"containers\":[{
            \"name\":\"fetch\",
            \"image\":\"busybox\",
            \"command\":[\"cat\",\"/data/residency/patched/${file}\"],
            \"volumeMounts\":[{\"name\":\"data\",\"mountPath\":\"/data\"}]
          }],
          \"volumes\":[{\"name\":\"data\",\"persistentVolumeClaim\":{\"claimName\":\"data-pvc\"}}]
        }
      }" 2>/dev/null | sed 's/pod ".*" deleted//g' > "$dest_dir/${file}" || true

    if [ -s "$dest_dir/${file}" ]; then
      echo "    Saved: ${dest_dir}/${file}"
    else
      rm -f "$dest_dir/${file}"
    fi
  done

  # f. Cleanup job
  oc delete job residency-experiment-patched --ignore-not-found=true 2>/dev/null
  while oc get pod -l variant=patched,experiment=ab-residency -o name 2>/dev/null | grep -q pod; do
    sleep 2
  done

  # g. Print rho_mt summary
  if [ -f "$dest_dir/summary.json" ] && command -v python3 &>/dev/null; then
    echo "  --- Residency summary ---"
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
per_tenant = d.get('per_tenant', {})
if not per_tenant:
    print('  (no per_tenant data in summary)')
    sys.exit(0)
residencies = {t: v.get('residency_token_seconds', 0)
               for t, v in per_tenant.items()}
rho = max(residencies.values()) / max(min(residencies.values()), 1e-9)
for t, r in sorted(residencies.items()):
    print(f'  {t}: {r:.2f} token-seconds')
print(f'  rho_mt = {rho:.3f}')
" "$dest_dir/summary.json" || echo "  (could not parse summary.json)"
  fi
  echo ""
}

# --- Main ---
echo "=== Asymmetric Fairness Experiment ==="
echo "  Duration:  ${DURATION}s"
echo "  Seed:      ${SEED}"
echo "  Rates:     ${RATES}"
echo "  Workload:  ${WORKLOAD}"
echo ""

# Parse rates into array
IFS=',' read -ra RATE_ARRAY <<< "$RATES"

# Clean up any leftover jobs from previous runs
echo "Cleaning up previous experiments..."
oc delete job -l variant=patched,experiment=ab-residency --ignore-not-found=true 2>/dev/null
while oc get pod -l variant=patched,experiment=ab-residency -o name 2>/dev/null | grep -q pod; do
  sleep 2
done
echo "  Done."
echo ""

# --- W5: Rate asymmetry sweep ---
if [[ "$WORKLOAD" == "both" || "$WORKLOAD" == "w5" ]]; then
  echo "========================================="
  echo "  W5: Rate Asymmetry (1:5 split)"
  echo "========================================="
  echo ""

  for rate in "${RATE_ARRAY[@]}"; do
    rate=$(echo "$rate" | xargs)  # trim whitespace
    echo "--- W5 | aggregate_rate=${rate} req/s ---"

    # Generate workload spec
    spec_file=$(mktemp /tmp/w5-spec-XXXXXX.yaml)
    gen_w5_spec "$rate" "$DURATION" "$SEED" > "$spec_file"

    dest_dir="$LOCAL_RESULTS/w5_sweep/agg${rate}"
    run_one "$spec_file" "$dest_dir" "$rate" "$DURATION" "$SEED"

    rm -f "$spec_file"
  done
fi

# --- W7: Prompt-length asymmetry sweep ---
if [[ "$WORKLOAD" == "both" || "$WORKLOAD" == "w7" ]]; then
  echo "========================================="
  echo "  W7: Prompt-Length Asymmetry (256 vs 4096)"
  echo "========================================="
  echo ""

  for rate in "${RATE_ARRAY[@]}"; do
    rate=$(echo "$rate" | xargs)  # trim whitespace
    echo "--- W7 | aggregate_rate=${rate} req/s ---"

    # Generate workload spec
    spec_file=$(mktemp /tmp/w7-spec-XXXXXX.yaml)
    gen_w7_spec "$rate" "$DURATION" "$SEED" > "$spec_file"

    dest_dir="$LOCAL_RESULTS/w7_sweep/agg${rate}"
    run_one "$spec_file" "$dest_dir" "$rate" "$DURATION" "$SEED"

    rm -f "$spec_file"
  done
fi

# --- Final summary ---
echo ""
echo "=== Experiment Complete ==="
echo "Results:"
if [[ "$WORKLOAD" == "both" || "$WORKLOAD" == "w5" ]]; then
  echo "  W5 (rate asymmetry):   $LOCAL_RESULTS/w5_sweep/"
fi
if [[ "$WORKLOAD" == "both" || "$WORKLOAD" == "w7" ]]; then
  echo "  W7 (prompt asymmetry): $LOCAL_RESULTS/w7_sweep/"
fi
echo ""
echo "Done."
