# Design Plan: Per-Tenant KV-Cache Residency Metric in vLLM

## 1. Objective

Instrument vLLM to expose a **Prometheus Counter** that accumulates per-tenant
KV-cache residency in **token-seconds** — the integral of resident KV tokens
over wall-clock time, with fair sharing of prefix-cached blocks across tenants.

```
vllm:residency_token_seconds_total{tenant_id="A"} → ∫ resident_tokens_A(t) dt
```

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  CLIENT                                                              │
│  POST /v1/chat/completions                                           │
│  { ..., "vllm_xargs": {"tenant_id": "tenant_A"} }                   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  API LAYER  (entrypoints/openai/)                                    │
│  vllm_xargs → SamplingParams.extra_args → Request.tenant_id          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  KV CACHE MANAGER  (v1/core/kv_cache_manager.py)                     │
│                                                                      │
│  Maintains incrementally:                                            │
│    holders: block_id → {tenant_id: refcount}                         │
│    tenant_resident_tokens: tenant_id → float                         │
│                                                                      │
│  Updated at:                                                         │
│    allocate_slots() → _residency_on_allocate()                       │
│    free()           → _residency_on_free()                           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ENGINE CORE  (v1/engine/core.py)                                    │
│                                                                      │
│  step():                                                             │
│    t₀ = time.monotonic()                                             │
│    scheduler.schedule()         ← blocks allocated here              │
│    model_executor.execute_model()                                    │
│    scheduler.update_from_output()  ← blocks freed here               │
│    dt = time.monotonic() - t₀                                        │
│                                                                      │
│    for tenant, tokens in tenant_resident_tokens.items():             │
│        counter.labels(tenant_id=tenant).inc(tokens * dt)             │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PROMETHEUS  /metrics endpoint                                       │
│  vllm:residency_token_seconds_total{tenant_id=...}                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Design: Incremental Residency Tracking

### Key idea

Instead of rebuilding a block→tenant map from scratch every step (O(N_req × B)),
we maintain residency state incrementally at the two sites where block ownership
changes: **allocation** and **free**. The per-step cost drops to O(N_tenants).

### Data structures (on KVCacheManager)

```python
# block_id → {tenant_id: request_refcount}
# Tracks which tenants hold each physical block.
# refcount handles intra-tenant sharing (multiple requests from same tenant
# sharing a prefix-cached block).
residency_holders: dict[int, dict[str, int]]

# tenant_id → current fractional token count
# Accounts for fair sharing: if k tenants share a block, each is credited
# block_size/k tokens. Updated on every allocate/free.
tenant_resident_tokens: dict[str, float]

# request_id → tenant_id (for free-time lookup)
req_to_tenant: dict[str, str]
```

### Why on KVCacheManager?

Both `allocate_slots()` and `free()` live on `KVCacheManager` and receive the
full `Request` object — so `request.tenant_id` is available without threading
anything new through the call chain.

---

## 4. Maintenance Logic

### On allocate (block B added to request R, tenant T)

```python
if B not in holders:
    holders[B] = {T: 1}
    tenant_resident_tokens[T] += block_size              # sole holder
elif T not in holders[B]:
    # Cross-tenant sharing: rebalance existing holders
    k = len(holders[B])
    for X in holders[B]:
        tenant_resident_tokens[X] -= block_size / k
        tenant_resident_tokens[X] += block_size / (k + 1)
    holders[B][T] = 1
    tenant_resident_tokens[T] += block_size / (k + 1)   # new holder
else:
    holders[B][T] += 1                                   # intra-tenant, no change
```

### On free (block B removed from request R, tenant T)

```python
holders[B][T] -= 1
if holders[B][T] == 0:
    k = len(holders[B])                  # includes T
    del holders[B][T]
    tenant_resident_tokens[T] -= block_size / k
    if not holders[B]:
        del holders[B]                   # block fully released
    else:
        # Rebalance remaining: 1/k → 1/(k-1)
        for X in holders[B]:
            tenant_resident_tokens[X] -= block_size / k
            tenant_resident_tokens[X] += block_size / (k - 1)
# else: tenant still holds via other requests, no change
```

### Conservation property

At any point in time:
```
∑_tenants tenant_resident_tokens[t] = total_allocated_non_null_blocks × block_size
```

Every block contributes exactly `block_size` total tokens across all its holders
(split as `block_size / k` each for k holders).

---

## 5. Step-Time Charging

### Location: `vllm/v1/engine/core.py` → `EngineCore.step()`

```python
def _accumulate_residency(self, dt: float) -> None:
    tenant_tokens = self.scheduler.kv_cache_manager.tenant_resident_tokens
    for tenant, tokens in tenant_tokens.items():
        if tokens > 0:
            self._residency_counter.labels(tenant_id=tenant).inc(tokens * dt)
```

This charges **all allocated blocks** — any block in GPU memory contributes to
residency regardless of whether its request was scheduled this step. This is
the correct memory occupancy accounting: those blocks consume GPU capacity.

### What dt includes

```
t₀ ── schedule() ── execute_model() ── update_from_output() ── t₁
      ↑ allocate                        ↑ free

dt = t₁ - t₀  (full step wall-clock time)
```

`dt` excludes the `_accumulate_residency()` overhead itself.

### Caveat: last-step under-count for finishing requests

Requests that complete during `update_from_output()` have their blocks freed
(and `tenant_resident_tokens` decremented) before `_accumulate_residency(dt)`
runs. This means those blocks were resident for the full step but aren't charged
for it — an under-count of `n_blocks × block_size × dt` per finishing request.

Over a multi-minute experiment this is negligible (one step of ~40ms per request
lifetime). If exact accounting is needed, snapshot `tenant_resident_tokens`
before `update_from_output()` and use the snapshot for charging.

---

## 6. Handling Shared Blocks (Prefix Caching)

When prefix caching is enabled, vLLM reuses KV blocks across requests with
common prefixes. The `residency_holders` map correctly handles this:

- **Intra-tenant sharing** (same tenant's requests share a block): the refcount
  increments but `tenant_resident_tokens` doesn't change — the tenant was
  already credited for the block.
- **Cross-tenant sharing** (different tenants share a prefix block): the
  rebalance logic adjusts all holders from `1/k` to `1/(k+1)` share, ensuring
  conservation.

In practice, cross-tenant sharing is unlikely when tenants use distinct prompts.
The accounting handles it correctly regardless.

---

## 7. Prometheus Counter Details

### Registration (in EngineCore.__init__)

```python
from prometheus_client import Counter

self._residency_counter = Counter(
    name="vllm:residency_token_seconds_total",
    documentation=(
        "Cumulative KV-cache residency in token-seconds per tenant. "
        "Measures the integral of resident KV-cache tokens over time, "
        "with fair splitting of prefix-cached shared blocks."
    ),
    labelnames=["tenant_id"],
)
```

### What appears at `/metrics`

```prometheus
# HELP vllm:residency_token_seconds_total Cumulative KV-cache residency in token-seconds per tenant.
# TYPE vllm:residency_token_seconds_total counter
vllm:residency_token_seconds_total{tenant_id="tenant_A"} 482315.7
vllm:residency_token_seconds_total{tenant_id="tenant_B"} 120472.3
vllm:residency_token_seconds_total{tenant_id="tenant_C"} 915638.1
```

### Useful PromQL queries

| Query | Meaning |
|-------|---------|
| `vllm:residency_token_seconds_total` | Raw cumulative per tenant |
| `rate(vllm:residency_token_seconds_total[1m])` | Avg resident tokens per tenant (instantaneous) |
| `rate(...{tenant_id="A"}[5m]) / sum(rate(...[5m]))` | Tenant A's share of total residency |
| `rate(...{tenant_id="A"}[5m]) / rate(...{tenant_id="B"}[5m])` | Relative A:B ratio |

---

## 8. Plugin vs. Patch

vLLM's `stat_logger_plugins` interface only receives pre-aggregated stats. It
has no access to per-request block allocations, step timing, or tenant identity.
The `general_plugins` entry point only runs initialization code and cannot hook
the allocate/free paths.

A source patch is the only viable approach.

---

## 9. Workload Design

### Setup

- 1 model (Llama-3.1-8B), 1 GPU
- 3 tenants, each identified by `tenant_id` passed via `vllm_xargs`
- **All tenants send identical workload**: same prompt length, same output
  length, same arrival rate
- Unique random prompts per request (fixed length but randomized content) —
  ensures no prefix cache hits, removing caching as a confounding variable
- Prefix caching enabled but never triggers (no repeated prefixes)
- Poisson inter-arrivals

### Configuration

| Parameter | Value |
|-----------|-------|
| Tenants | 3 |
| Prompt tokens | 1024 |
| Max output tokens | 128 |
| Rate per tenant | 2 req/s (Poisson) |
| Duration | 5 minutes (300 requests/tenant) |
| Model | Llama-3.1-8B |
| KV blocks per request | ceil(1024 / 16) = 64 |

### Client driver

A Python script sends identical workload from each tenant, differing only in
`tenant_id`. Synthetic fixed-length prompts give deterministic KV block counts.

```python
import openai, time, random, threading

client = openai.OpenAI(base_url="http://cluster:8000/v1", api_key="dummy")

PROMPT_LEN = 1024
MAX_TOKENS = 128
RATE = 2.0  # req/s per tenant
TENANTS = ["tenant_A", "tenant_B", "tenant_C"]

def make_prompt(n_tokens):
    # Unique each time -> no prefix cache hits
    vocab = ["alpha", "beta", "gamma", "delta", "echo"]
    return " ".join(random.choices(vocab, k=n_tokens))

def send_for_tenant(tenant_id):
    while True:
        client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B",
            messages=[{"role": "user", "content": make_prompt(PROMPT_LEN)}],
            max_tokens=MAX_TOKENS,
            extra_body={"vllm_xargs": {"tenant_id": tenant_id}},
        )
        time.sleep(random.expovariate(RATE))  # Poisson

for tenant in TENANTS:
    threading.Thread(target=send_for_tenant, args=(tenant,)).start()
```

### What to measure / observe

1. **Residency distribution**: Under symmetric load, is residency actually
   equal across tenants? Or does vLLM's FCFS scheduling + continuous batching
   create systematic skew?
2. **Temporal dynamics**: How does per-tenant residency evolve over time?
   Are there transient unfairness spikes (e.g., one tenant's requests
   dominating the batch)?
3. **Conservation**: `sum(rate(residency[*]))` should approximate
   `total_kv_blocks_in_use × block_size` at steady state (validates metric
   correctness)
4. **Preemption effects**: Under memory pressure, does preemption
   disproportionately affect certain tenants?

---

## 10. Files Modified

| File | Change |
|------|--------|
| `vllm/v1/request.py` | Add `self.tenant_id` field from `extra_args` (~4 lines) |
| `vllm/v1/core/kv_cache_manager.py` | Add residency state, `_residency_on_allocate()`, `_residency_on_free()` (~60 lines) |
| `vllm/v1/engine/core.py` | Add timing in `step()`, `_accumulate_residency()`, register Counter (~15 lines) |

Total patch: ~80 lines of new code.

---

## 11. Performance Impact

| Operation | Cost | When |
|-----------|------|------|
| `_residency_on_allocate` | O(new_blocks) per request | At allocation (decode: 0-1 blocks/step) |
| `_residency_on_free` | O(blocks_per_request) | Once per request lifetime |
| Rebalance (cross-tenant sharing) | O(k) per shared block | Rare |
| **`_accumulate_residency`** | **O(N_tenants)** | **Once per step** |

In steady-state decode with block_size=16: a new block is allocated every 16
steps per request. The per-step overhead is O(N_tenants) for the charging loop
(~3-5 dict reads), plus amortized O(1) for allocation bookkeeping.

**Total overhead: <0.1% of step time.** Negligible compared to GPU execution.

---

## 12. Deployment

```bash
# On cluster node
git clone <fork-with-residency-patch> && cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .

# Launch server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --port 8000

# Client sends requests with tenant_id
curl http://cluster:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "Hello"}],
    "vllm_xargs": {"tenant_id": "tenant_A"}
  }'

# Prometheus scrapes
curl http://cluster:8000/metrics | grep residency
```

---

## 13. Correctness Invariants

1. **Conservation**: `∑ tenant_resident_tokens[t] = allocated_blocks × block_size`
2. **Monotonicity**: Prometheus counter only increases
3. **Fair splitting**: Shared blocks split equally among holding tenants
4. **Completeness**: Every allocated block contributes to some tenant's residency
5. **Incremental consistency**: `tenant_resident_tokens` is always in sync with
   `residency_holders` — updated atomically at each allocate/free
6. **dt excludes accounting**: Timer stops before `_accumulate_residency()` runs
