# Deployment Plan: Patched vLLM on Kubernetes

## Overview

We overlay 3 patched Python files onto the official `vllm/vllm-openai` Docker
image. No compilation needed — the patch is pure Python and the base image has
all CUDA/C++ extensions pre-built.

---

## Repository Structure

```
residency-vllm/
├── Dockerfile
├── .github/workflows/docker-build-push.yml
├── k8s/
│   └── deployment.yaml
├── docs/
│   └── ...
└── vllm/v1/
    ├── request.py                  ← add tenant_id field
    ├── engine/
    │   └── core.py                ← step timing + _accumulate_residency()
    └── core/
        └── kv_cache_manager.py    ← residency_holders + tenant_resident_tokens
```

Only the 3 modified files live in `vllm/v1/`. The `cp -r` overlay replaces just
these files in the installed vLLM package; everything else stays from the base.

---

## Dockerfile

```dockerfile
ARG VLLM_BASE_VERSION=v0.15.1
FROM vllm/vllm-openai:${VLLM_BASE_VERSION}

# prometheus_client is already in the base image — no extra pip install needed

# Find where vLLM is installed
RUN VLLM_LOCATION=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))") && \
    echo "$VLLM_LOCATION" > /tmp/vllm_location.txt

# Copy only the patched files
COPY vllm/ /tmp/vllm-patch/

# Overlay patched files onto the installed package
RUN VLLM_LOCATION=$(cat /tmp/vllm_location.txt) && \
    cp -r /tmp/vllm-patch/* "$VLLM_LOCATION/" && \
    rm -rf /tmp/vllm-patch

ENTRYPOINT ["vllm", "serve"]
```

### How the overlay works

`cp -r` merges directory trees. If the base image has:
```
/usr/lib/python3/site-packages/vllm/v1/request.py  (original)
```

And we copy our `vllm/v1/request.py` on top, only that file gets replaced.
All other files (including compiled `.so` extensions) remain untouched.

---

## GitHub Actions Workflow

```yaml
# .github/workflows/docker-build-push.yml
name: Build and Push Docker Image

on:
  push:
    tags:
      - "v*"
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=sha,prefix=sha-

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          build-args: |
            VLLM_BASE_VERSION=v0.15.1
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Triggering a build

```bash
git tag v0.15.1-residency
git push origin v0.15.1-residency
# → GitHub Actions builds and pushes ghcr.io/inference-sim/residency-vllm:0.15.1-residency
```

---

## Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-residency
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-residency
  template:
    metadata:
      labels:
        app: vllm-residency
    spec:
      containers:
        - name: vllm
          image: ghcr.io/inference-sim/residency-vllm:0.15.1-residency
          imagePullPolicy: IfNotPresent
          resources:
            limits:
              nvidia.com/gpu: 1
          args:
            - "--model"
            - "meta-llama/Llama-3.1-8B"
            - "--port"
            - "8000"
          ports:
            - containerPort: 8000
              name: http
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
            - name: HOME
              value: "/cache"
            - name: HF_HOME
              value: "/cache/huggingface"
          volumeMounts:
            - name: cache
              mountPath: /cache
      volumes:
        - name: cache
          persistentVolumeClaim:
            claimName: vllm-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-residency
spec:
  selector:
    app: vllm-residency
  ports:
    - port: 8000
      targetPort: 8000
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-cache-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
```

---

## End-to-End Flow

```bash
# 1. Write the patched files (request.py, core.py, kv_cache_manager.py)
#    into vllm/v1/ in this repo

# 2. Tag and push → triggers CI build
git tag v0.15.1-residency && git push origin v0.15.1-residency

# 3. Deploy on cluster
kubectl apply -f k8s/deployment.yaml

# 4. Wait for pod ready
kubectl wait --for=condition=ready pod -l app=vllm-residency --timeout=300s

# 5. Run workload (from client machine or another pod)
python client_driver.py

# 6. Scrape residency metrics
curl http://vllm-residency:8000/metrics | grep residency_token_seconds
```

---

## Notes

- **Base version**: `v0.15.1` is the latest stable vLLM release. The v1 engine
  (with the scheduler and KVCacheManager we're patching) is the default in this
  version.
- **No compilation**: The overlay only touches Python files. Build time is
  seconds (just `COPY` + `cp`), not the 30+ minutes a full vLLM build takes.
- **Prometheus**: vLLM already serves `/metrics` on the API port (8000). The
  residency counter appears there alongside existing vLLM metrics. No sidecar
  or additional exporter needed.
- **Model download**: First pod startup downloads Llama-3.1-8B to the PVC cache.
  Subsequent restarts reuse the cached weights.
