#!/bin/bash
# run_ab_experiment.sh — A/B comparison: patched vs vanilla vLLM
# Usage: ./run_ab_experiment.sh [--duration 300] [--rate 2.0] [--seed 42]
#
# Runs identical Poisson workloads against both servers simultaneously,
# downloads results, and cleans up.

set -euo pipefail

DURATION="${DURATION:-300}"
RATE="${RATE:-2.0}"
SEED="${SEED:-42}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --duration) DURATION="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_YAML="$SCRIPT_DIR/k8s/ab-experiment.yaml"
LOCAL_RESULTS="$SCRIPT_DIR/results"

echo "=== A/B Residency Experiment ==="
echo "  Duration: ${DURATION}s"
echo "  Rate:     ${RATE} req/s per tenant"
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

# --- Apply both Jobs ---
echo "Launching A/B experiment Jobs..."
oc apply -f "$JOB_YAML"

# --- Patch parameters if non-default ---
for VARIANT in patched vanilla; do
  JOB_NAME="residency-experiment-${VARIANT}"
  DRIVER_ARGS="echo \"Waiting for vLLM server to be ready...\"
python3 -c \"
import urllib.request, time
while True:
    try:
        urllib.request.urlopen('http://localhost:8000/v1/models')
        break
    except Exception:
        time.sleep(5)
\"
echo \"Server ready. Starting experiment (${VARIANT} variant).\"
python3 workload_driver.py \\
  --base-url http://localhost:8000 \\
  --model \"Qwen/Qwen3-14B\" \\
  --tenants \"tenant_A,tenant_B,tenant_C\" \\
  --rate ${RATE} \\
  --duration ${DURATION} \\
  --prompt-tokens 1024 \\
  --max-tokens 128 \\
  --seed ${SEED} \\
  --output-dir /data/residency/${VARIANT}
echo \"Experiment complete. Results written to /data/residency/${VARIANT}/\"
touch /data/residency/${VARIANT}/.done"

  if [ "$DURATION" != "300" ] || [ "$RATE" != "2.0" ] || [ "$SEED" != "42" ]; then
    oc patch job "$JOB_NAME" --type='json' \
      -p="[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/1/args/0\",\"value\":\"$DRIVER_ARGS\"}]" \
      2>/dev/null || true
  fi
done

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
    }" > "$LOCAL_RESULTS/${VARIANT}/summary.json" 2>/dev/null

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
    }" > "$LOCAL_RESULTS/${VARIANT}/requests.csv" 2>/dev/null

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
