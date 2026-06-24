#!/bin/bash
# run_ab_experiment.sh — A/B comparison: patched vs vanilla vLLM
# Usage: ./run_ab_experiment.sh [--duration 300] [--rate 2.0] [--seed 42] [--tenants "a,b,c,d,e"]
#
# Generates a blis observe workload spec from the arguments, uploads it to
# the shared PVC, launches both experiment Jobs, downloads results (including
# trace files), and cleans up.

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
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_YAML="$REPO_ROOT/k8s/ab-experiment.yaml"
LOCAL_RESULTS="$REPO_ROOT/results"

# Parse tenants into array
IFS=',' read -ra TENANT_ARRAY <<< "$TENANTS"
NUM_TENANTS=${#TENANT_ARRAY[@]}

# Compute aggregate rate and horizon
AGGREGATE_RATE=$(echo "$RATE * $NUM_TENANTS" | bc -l)
HORIZON=$(echo "$DURATION * 1000000" | bc)
RATE_FRACTION=$(echo "scale=6; 1.0 / $NUM_TENANTS" | bc -l)

echo "=== A/B Residency Experiment (blis observe) ==="
echo "  Duration:       ${DURATION}s"
echo "  Rate:           ${RATE} req/s per tenant"
echo "  Aggregate rate: ${AGGREGATE_RATE} req/s"
echo "  Tenants:        ${NUM_TENANTS} (${TENANTS})"
echo "  Seed:           ${SEED}"
echo ""

# --- Generate workload spec YAML ---
echo "Generating workload spec..."
WORKLOAD_SPEC=$(mktemp /tmp/workload-XXXXXX.yaml)

cat > "$WORKLOAD_SPEC" <<EOF
version: "2"
seed: ${SEED}
aggregate_rate: ${AGGREGATE_RATE}
horizon: ${HORIZON}

clients:
EOF

for TENANT in "${TENANT_ARRAY[@]}"; do
  TENANT=$(echo "$TENANT" | xargs)  # trim whitespace
  cat >> "$WORKLOAD_SPEC" <<EOF
  - id: "${TENANT}"
    tenant_id: "${TENANT}"
    rate_fraction: ${RATE_FRACTION}
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
done

echo "  Generated: $WORKLOAD_SPEC"

# --- Teardown previous runs ---
echo "Cleaning up previous A/B experiments..."
oc delete job -l experiment=ab-residency --ignore-not-found=true 2>/dev/null
while oc get pod -l experiment=ab-residency -o name 2>/dev/null | grep -q pod; do
  sleep 2
done
echo "  Done."

# --- Upload workload spec + clear stale signal files ---
echo "Uploading workload spec to data-pvc..."
oc run ab-setup --rm -i --restart=Never --image=busybox \
  --overrides='{
    "spec":{
      "containers":[{
        "name":"ab-setup",
        "image":"busybox",
        "command":["sh","-c","mkdir -p /data/residency && chmod 777 /data/residency && rm -rf /data/residency/patched /data/residency/vanilla && cat > /data/residency/workload.yaml"],
        "stdin": true,
        "securityContext":{"runAsUser":0},
        "volumeMounts":[{"name":"data","mountPath":"/data"}]
      }],
      "volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc"}}]
    }
  }' < "$WORKLOAD_SPEC" 2>/dev/null || true

rm -f "$WORKLOAD_SPEC"
echo "  Done."

# --- Apply both Jobs (substitute runtime parameters into YAML) ---
echo "Launching A/B experiment Jobs..."
sed \
  -e "s|--rate 2.0|--rate ${RATE}|g" \
  -e "s|--duration 300|--duration ${DURATION}|g" \
  -e "s|--seed 42|--seed ${SEED}|g" \
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

# Files to download from each variant
RESULT_FILES="summary.json requests.csv trace.yaml trace.csv trace.itl.csv"

for VARIANT in patched vanilla; do
  echo "  Fetching ${VARIANT} results..."

  for FILE in $RESULT_FILES; do
    # Use filename stem for pod name (must be DNS-safe)
    POD_SUFFIX=$(echo "${FILE}" | tr '.' '-')
    oc run "fetch-${VARIANT}-${POD_SUFFIX}" --rm -i --restart=Never --image=busybox \
      --overrides="{
        \"spec\":{
          \"containers\":[{
            \"name\":\"fetch\",
            \"image\":\"busybox\",
            \"command\":[\"cat\",\"/data/residency/${VARIANT}/${FILE}\"],
            \"volumeMounts\":[{\"name\":\"data\",\"mountPath\":\"/data\"}]
          }],
          \"volumes\":[{\"name\":\"data\",\"persistentVolumeClaim\":{\"claimName\":\"data-pvc\"}}]
        }
      }" 2>/dev/null | sed 's/pod ".*" deleted//g' > "$LOCAL_RESULTS/${VARIANT}/${FILE}" || true

    if [ -s "$LOCAL_RESULTS/${VARIANT}/${FILE}" ]; then
      echo "    Saved: $LOCAL_RESULTS/${VARIANT}/${FILE}"
    else
      # File may not exist (e.g., trace.itl.csv if --record-itl failed); remove empty stub
      rm -f "$LOCAL_RESULTS/${VARIANT}/${FILE}"
    fi
  done
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
import json
with open('$LOCAL_RESULTS/${VARIANT}/summary.json') as f:
    d = json.load(f)
# Support both old format (per_tenant.X.ttft_ms.p50) and new (overall.latency.ttft_ms.median)
if 'overall' in d:
    lat = d['overall'].get('latency', {})
    ttft = lat.get('ttft_ms', {}).get('median', 'N/A')
    itl = lat.get('itl_ms', {}).get('median', 'N/A')
    e2e = lat.get('e2e_ms', {}).get('median', 'N/A')
    tput = d['overall'].get('throughput', {}).get('output_tokens_per_sec', 'N/A')
    print(f'  $LABEL: TTFT={ttft}ms  ITL={itl}ms  E2E={e2e}ms  out_tok/s={tput}')
else:
    tenants = d.get('per_tenant', {})
    first = next(iter(tenants.values()), {})
    ttft = first.get('ttft_ms', {}).get('p50', first.get('ttft_ms', {}).get('median', 'N/A'))
    itl = first.get('itl_ms', {}).get('p50', first.get('itl_ms', {}).get('median', 'N/A'))
    e2e = first.get('e2e_ms', {}).get('p50', first.get('e2e_ms', {}).get('median', 'N/A'))
    n = sum(t.get('num_requests', 0) for t in tenants.values())
    print(f'  $LABEL: TTFT={ttft}ms  ITL={itl}ms  E2E={e2e}ms  reqs={n}')
" 2>/dev/null || echo "  ${VARIANT}: (could not parse summary)"
    fi
  done
fi
echo ""
echo "Done."
