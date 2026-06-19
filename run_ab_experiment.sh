#!/bin/bash
# run_ab_experiment.sh — A/B comparison: patched vs vanilla vLLM
# Usage: ./run_ab_experiment.sh [--duration 300] [--rate 2.0] [--seed 42] [--tenants "a,b,c,d,e"]
#
# Runs identical Poisson workloads against both servers simultaneously,
# downloads results, and cleans up.

set -euo pipefail

DURATION="${DURATION:-300}"
RATE="${RATE:-2.0}"
SEED="${SEED:-42}"
TENANTS="${TENANTS:-tenant_A,tenant_B,tenant_C,tenant_D,tenant_E}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --duration) DURATION="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --tenants) TENANTS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_YAML="$SCRIPT_DIR/k8s/ab-experiment.yaml"
LOCAL_RESULTS="$SCRIPT_DIR/results"

NUM_TENANTS=$(echo "$TENANTS" | tr ',' '\n' | wc -l | tr -d ' ')

echo "=== A/B Residency Experiment ==="
echo "  Duration: ${DURATION}s"
echo "  Rate:     ${RATE} req/s per tenant"
echo "  Tenants:  ${NUM_TENANTS} (${TENANTS})"
echo "  Seed:     ${SEED}"
echo ""

# --- Teardown previous runs ---
echo "Cleaning up previous A/B experiments..."
oc delete job -l experiment=ab-residency --ignore-not-found=true 2>/dev/null
while oc get pod -l experiment=ab-residency -o name 2>/dev/null | grep -q pod; do
  sleep 2
done
echo "  Done."

# --- Clear stale signal files ---
oc run ab-cleanup --rm -i --restart=Never --image=busybox \
  --overrides='{
    "spec":{
      "containers":[{
        "name":"ab-cleanup",
        "image":"busybox",
        "command":["sh","-c","rm -f /data/residency/patched/.done /data/residency/vanilla/.done"],
        "volumeMounts":[{"name":"data","mountPath":"/data"}]
      }],
      "volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc"}}]
    }
  }' 2>/dev/null || true

# --- Apply both Jobs (substitute runtime parameters into YAML) ---
echo "Launching A/B experiment Jobs..."
sed \
  -e "s|--rate 2.0|--rate ${RATE}|g" \
  -e "s|--duration 300|--duration ${DURATION}|g" \
  -e "s|--seed 42|--seed ${SEED}|g" \
  -e "s|tenant_A,tenant_B,tenant_C,tenant_D,tenant_E|${TENANTS}|g" \
  "$JOB_YAML" | oc apply -f -

echo "  Jobs created. Pods scheduling..."
echo ""

# --- Wait for both Jobs to complete ---
echo "Waiting for both experiments to finish..."
echo "(Following patched driver logs — vanilla runs in parallel)"
echo "---"

# Wait for pods to become ready before tailing logs
oc wait --for=condition=ready pod -l variant=patched,experiment=ab-residency --timeout=600s 2>/dev/null || true
oc logs -f job/residency-experiment-patched -c driver 2>/dev/null || true

echo "---"
echo "Patched variant complete. Checking vanilla..."

# Wait for vanilla to finish (may already be done)
oc wait --for=condition=complete job/residency-experiment-vanilla --timeout=1200s 2>/dev/null || true
echo "Vanilla variant complete."
echo ""

# --- Download results ---
echo "=== Downloading Results ==="
mkdir -p "$LOCAL_RESULTS/patched" "$LOCAL_RESULTS/vanilla"

for VARIANT in patched vanilla; do
  echo "  Fetching ${VARIANT} results..."

  oc run "fetch-${VARIANT}-summary" --rm -i --restart=Never --image=busybox \
    --overrides="{
      \"spec\":{
        \"containers\":[{
          \"name\":\"fetch-${VARIANT}-summary\",
          \"image\":\"busybox\",
          \"command\":[\"cat\",\"/data/residency/${VARIANT}/summary.json\"],
          \"volumeMounts\":[{\"name\":\"data\",\"mountPath\":\"/data\"}]
        }],
        \"volumes\":[{\"name\":\"data\",\"persistentVolumeClaim\":{\"claimName\":\"data-pvc\"}}]
      }
    }" 2>/dev/null | sed '/^pod.*deleted$/d' > "$LOCAL_RESULTS/${VARIANT}/summary.json"

  oc run "fetch-${VARIANT}-csv" --rm -i --restart=Never --image=busybox \
    --overrides="{
      \"spec\":{
        \"containers\":[{
          \"name\":\"fetch-${VARIANT}-csv\",
          \"image\":\"busybox\",
          \"command\":[\"cat\",\"/data/residency/${VARIANT}/requests.csv\"],
          \"volumeMounts\":[{\"name\":\"data\",\"mountPath\":\"/data\"}]
        }],
        \"volumes\":[{\"name\":\"data\",\"persistentVolumeClaim\":{\"claimName\":\"data-pvc\"}}]
      }
    }" 2>/dev/null | sed '/^pod.*deleted$/d' > "$LOCAL_RESULTS/${VARIANT}/requests.csv"

  echo "    Saved: $LOCAL_RESULTS/${VARIANT}/summary.json"
  echo "    Saved: $LOCAL_RESULTS/${VARIANT}/requests.csv"
done

echo ""

# --- Cleanup ---
echo "Cleaning up Jobs..."
oc delete job -l experiment=ab-residency --ignore-not-found=true
echo ""

# --- Summary ---
echo "=== A/B Experiment Complete ==="
echo "Results:"
echo "  Patched: $LOCAL_RESULTS/patched/"
echo "  Vanilla: $LOCAL_RESULTS/vanilla/"
echo ""
echo "Quick comparison:"
if command -v python3 &>/dev/null; then
  for VARIANT in patched vanilla; do
    if [ -f "$LOCAL_RESULTS/${VARIANT}/summary.json" ]; then
      LABEL=$(echo "$VARIANT" | tr '[:lower:]' '[:upper:]')
      python3 -c "
import json, sys
with open('$LOCAL_RESULTS/${VARIANT}/summary.json') as f:
    d = json.load(f)
tenants = d.get('per_tenant', {})
first = next(iter(tenants.values()), {})
ttft = first.get('ttft_ms', {}).get('p50', 'N/A')
itl = first.get('itl_ms', {}).get('p50', 'N/A')
e2e = first.get('e2e_latency_ms', {}).get('p50', 'N/A')
print(f'  $LABEL: TTFT p50={ttft}ms  ITL p50={itl}ms  E2E p50={e2e}ms')
" 2>/dev/null || echo "  ${VARIANT}: (could not parse summary)"
    fi
  done
fi
echo ""
echo "Done."
