# residency-vllm

Per-tenant KV-cache residency instrumentation for vLLM. Exposes a Prometheus
counter (`vllm:residency_token_seconds_total`) that accumulates token-seconds
of GPU KV-cache occupancy per tenant, with fair splitting of shared blocks.

## How it works

Three patched Python files are overlaid onto the official `vllm/vllm-openai`
Docker image via `cp -r`. No compilation required — the base image ships all
CUDA/C++ extensions pre-built.

```
vllm/v1/
├── request.py              ← extracts tenant_id from vllm_xargs
├── engine/core.py          ← step timing + _accumulate_residency()
└── core/kv_cache_manager.py ← residency_holders + tenant_resident_tokens
```

## Prerequisites

- An OpenShift cluster with GPU nodes (nvidia.com/gpu)
- `oc` CLI configured and logged in (`oc login`)
- A Hugging Face token with access to `meta-llama/Llama-3.1-8B`

## Quick start

### 1. Create the HF token secret

```bash
oc create secret generic hf-token --from-literal=token=<YOUR_HF_TOKEN>
```

### 2. Deploy

```bash
oc apply -f k8s/deployment.yaml
oc wait --for=condition=ready pod -l app=vllm-residency --timeout=600s
```

### 3. Send a request with tenant_id

```bash
# From within the cluster, or via port-forward:
oc port-forward svc/vllm-residency 8000:8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "extra_body": {"vllm_xargs": {"tenant_id": "tenant_A"}}
  }'
```

### 4. Scrape residency metrics

```bash
curl http://localhost:8000/metrics | grep residency_token_seconds
```

Output:
```
vllm:residency_token_seconds_total{tenant_id="tenant_A"} 482315.7
```

### 5. Run the workload driver

```bash
pip install -r requirements-client.txt
python workload_driver.py --base-url http://localhost:8000 --duration 300 --rate 2.0
```

Results are written to `./results/summary.json` and `./results/requests.csv`.

## Building the image

CI builds automatically on version tags:

```bash
git tag v0.23.0-residency
git push origin v0.23.0-residency
```

This triggers GitHub Actions to build and push to
`ghcr.io/inference-sim/residency-vllm:<version>`.

To build locally:

```bash
docker build -t residency-vllm:latest .
```

## Repository layout

```
.
├── Dockerfile                           # Overlay patch onto vllm-openai base
├── .github/workflows/docker-build-push.yml  # CI: build + push to GHCR
├── k8s/deployment.yaml                  # Deployment + Service + PVC
├── workload_driver.py                   # Poisson-arrival experiment runner
├── requirements-client.txt              # Client-side dependencies (aiohttp)
├── vllm/v1/                             # Patched files (overlay source)
│   ├── request.py
│   ├── engine/core.py
│   └── core/kv_cache_manager.py
└── docs/
    ├── vllm-residency-design.md         # Full design document
    ├── vllm-patch-spec.md               # Exact patch specification
    └── deployment-plan.md               # Detailed deployment walkthrough
```

## Configuration

| Environment variable | Purpose |
|---------------------|---------|
| `HF_TOKEN` | Hugging Face access token (via k8s secret) |
| `HF_HOME` | Model cache directory (defaults to `/cache/huggingface`) |

Server arguments are passed via the container's `args` in the deployment yaml.
Default: `--model meta-llama/Llama-3.1-8B --port 8000`.

## Useful PromQL queries

| Query | Meaning |
|-------|---------|
| `vllm:residency_token_seconds_total` | Raw cumulative per tenant |
| `rate(vllm:residency_token_seconds_total[1m])` | Avg resident tokens (instantaneous) |
| `rate(...{tenant_id="A"}[5m]) / sum(rate(...[5m]))` | Tenant A's share of total |

## Design docs

- [Design document](docs/vllm-residency-design.md) — architecture, incremental tracking, conservation invariants
- [Patch specification](docs/vllm-patch-spec.md) — exact code changes with line numbers
- [Deployment plan](docs/deployment-plan.md) — end-to-end deployment walkthrough
