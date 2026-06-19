#!/bin/bash
# run_experiment.sh — Launch a residency experiment on OpenShift
# Usage: ./run_experiment.sh [--duration 300] [--rate 2.0] [--no-wait]
#
# Cleans up any previous experiment, launches the Job, waits for
# completion, downloads results locally, and tears down the pod.
# Pass --no-wait to run in background without downloading results.

set -euo pipefail

DURATION="${DURATION:-300}"
RATE="${RATE:-2.0}"
WAIT=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --duration) DURATION="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --no-wait) WAIT=false; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_YAML="$SCRIPT_DIR/k8s/experiment-pod.yaml"

echo "=== Residency Experiment ==="
echo "  Duration: ${DURATION}s"
echo "  Rate:     ${RATE} req/s per tenant"
echo ""

# --- Teardown previous run ---
echo "Cleaning up previous experiment..."
oc delete job residency-experiment --ignore-not-found=true 2>/dev/null
# Wait for pod to be fully gone
while oc get pod -l job-name=residency-experiment -o name 2>/dev/null | grep -q pod; do
  sleep 2
done
echo "  Done."

# --- Clear stale signal file ---
# (In case a previous run left .done on the PVC)
oc run cleanup --rm -i --restart=Never --image=busybox -- \
  sh -c "rm -f /data/residency/.done" \
  --overrides='{"spec":{"containers":[{"name":"cleanup","image":"busybox","command":["sh","-c","rm -f /data/residency/.done"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc"}}]}}' \
  2>/dev/null || true

# --- Apply with parameter overrides ---
echo "Launching experiment Job..."
oc apply -f "$JOB_YAML"

# Patch in runtime parameters if different from defaults
if [ "$DURATION" != "300" ] || [ "$RATE" != "2.0" ]; then
  DRIVER_ARGS="echo \"Waiting for vLLM server to be ready...\"
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
  sleep 5
done
echo \"Server ready. Starting experiment.\"
python3 workload_driver.py \\
  --base-url http://localhost:8000 \\
  --model \"Qwen/Qwen3-14B\" \\
  --tenants \"tenant_A,tenant_B,tenant_C\" \\
  --rate ${RATE} \\
  --duration ${DURATION} \\
  --prompt-tokens 1024 \\
  --max-tokens 128 \\
  --output-dir /data/residency
echo \"Experiment complete. Results written to /data/residency/\"
touch /data/residency/.done"

  oc patch job residency-experiment --type='json' \
    -p="[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/1/args/0\",\"value\":\"$DRIVER_ARGS\"}]" \
    2>/dev/null || true
fi

echo "  Job created. Pod scheduling..."
oc wait --for=condition=ready pod -l job-name=residency-experiment --timeout=600s 2>/dev/null || true
echo ""

if [ "$WAIT" = true ]; then
  echo "Following driver logs (Ctrl+C to detach — experiment continues):"
  echo "---"
  oc logs -f job/residency-experiment -c driver 2>/dev/null || true

  echo ""
  echo "=== Experiment Complete ==="

  # Download results locally
  LOCAL_RESULTS="$SCRIPT_DIR/results"
  mkdir -p "$LOCAL_RESULTS"
  echo "Downloading results from data-pvc..."
  oc run fetch-results --rm -i --restart=Never --image=busybox \
    --overrides='{
      "spec":{
        "containers":[{
          "name":"fetch-results",
          "image":"busybox",
          "command":["cat","/data/residency/summary.json"],
          "volumeMounts":[{"name":"data","mountPath":"/data"}]
        }],
        "volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc"}}]
      }
    }' 2>/dev/null | sed '/^pod.*deleted$/d' > "$LOCAL_RESULTS/summary.json"

  oc run fetch-csv --rm -i --restart=Never --image=busybox \
    --overrides='{
      "spec":{
        "containers":[{
          "name":"fetch-csv",
          "image":"busybox",
          "command":["cat","/data/residency/requests.csv"],
          "volumeMounts":[{"name":"data","mountPath":"/data"}]
        }],
        "volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc"}}]
      }
    }' 2>/dev/null | sed '/^pod.*deleted$/d' > "$LOCAL_RESULTS/requests.csv"

  echo "  Saved: $LOCAL_RESULTS/summary.json"
  echo "  Saved: $LOCAL_RESULTS/requests.csv"
  echo ""

  echo "Cleaning up Job..."
  oc delete job residency-experiment --ignore-not-found=true
  echo "Done."
else
  echo "Experiment running in background."
  echo ""
  echo "  Watch:   oc logs -f job/residency-experiment -c driver"
  echo "  Status:  oc get pod -l job-name=residency-experiment"
  echo "  Results: stored on data-pvc at /residency/"
  echo ""
  echo "The Job auto-deletes 60s after completion (ttlSecondsAfterFinished)."
fi
