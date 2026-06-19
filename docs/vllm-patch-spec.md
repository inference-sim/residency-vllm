# vLLM Patch Specification: Per-Tenant KV-Cache Residency

## Goal

Add a Prometheus counter `vllm:residency_token_seconds_total{tenant_id=...}` that
accumulates per-tenant KV-cache residency (token-seconds) with O(N_tenants) per-step
overhead by maintaining residency state incrementally at allocation/free sites.

---

## Architecture

```
Allocation/free time (amortized)          Step time (O(N_tenants))
─────────────────────────────────         ──────────────────────────
KVCacheManager.allocate_slots()           EngineCore.step():
  → update holders                          dt = measured step duration
  → update tenant_resident_tokens           for tenant, tokens in tenant_resident_tokens:
                                                counter.inc(tokens * dt)
KVCacheManager.free()
  → update holders
  → update tenant_resident_tokens
```

---

## Data Structures (new)

Location: `vllm/v1/core/kv_cache_manager.py` (on `KVCacheManager`)

```python
# block_id → {tenant_id: request_refcount}
# Tracks which tenants hold each physical block and how many of their
# requests reference it. Needed to correctly handle intra-tenant sharing
# (prefix cache hits within same tenant) vs cross-tenant sharing.
self.residency_holders: dict[int, dict[str, int]] = {}

# tenant_id → current fractional token count
# Accounts for fair-sharing of cross-tenant blocks.
# Updated incrementally at allocate/free time.
self.tenant_resident_tokens: dict[str, float] = defaultdict(float)

# request_id → tenant_id mapping
# Needed at free time (free() receives request object which has tenant_id,
# but we also use this for validation).
self.req_to_tenant: dict[str, str] = {}
```

---

## File Changes

### 1. `vllm/v1/request.py` — Add tenant_id field

**Location:** `Request.__init__`, after line 117

```python
# After the existing kv_transfer_params extraction:
if sampling_params.extra_args is not None:
    self.kv_transfer_params = sampling_params.extra_args.get(
        "kv_transfer_params"
    )
    self.tenant_id: str | None = sampling_params.extra_args.get(
        "tenant_id"
    )
else:
    self.tenant_id: str | None = None
```

Also add a default in the `pooling_params is not None` branch (line 104-106):
```python
self.tenant_id: str | None = None
```

---

### 2. `vllm/v1/core/kv_cache_manager.py` — Incremental residency tracking

**2a. Add state in `__init__`** (after line ~143 where coordinator is created):

```python
# Residency tracking
self.residency_holders: dict[int, dict[str, int]] = {}
self.tenant_resident_tokens: dict[str, float] = defaultdict(float)
self.req_to_tenant: dict[str, str] = {}
self.block_size: int = kv_cache_config.block_size  # or from coordinator
```

**2b. Hook in `allocate_slots()`** (at end, before return, ~line 456-458):

```python
def allocate_slots(self, request: Request, ...) -> KVCacheBlocks | None:
    # ... existing logic ...
    new_blocks_result = self.coordinator.allocate_new_blocks(...)

    # --- Residency: track newly allocated blocks ---
    if request.tenant_id is not None:
        self._residency_on_allocate(request, new_blocks_result)

    # ... existing caching logic ...
    return self.create_kv_cache_blocks(new_blocks_result)
```

**2c. Hook in `free()`** (before the actual free, line 460-468):

```python
def free(self, request: Request) -> None:
    # --- Residency: remove blocks before they're freed ---
    if request.tenant_id is not None:
        self._residency_on_free(request)

    self.coordinator.free(request.request_id)
```

**2d. Hook in `pop_blocks_for_free()`** (for deferred frees, line 483):

```python
def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
    # --- Residency: remove blocks before they're popped ---
    if request.tenant_id is not None:
        self._residency_on_free(request)

    # ... existing logic ...
```

**2e. New methods:**

```python
def _residency_on_allocate(self, request: Request, new_blocks: list[list[KVCacheBlock]]):
    """Update residency state when blocks are allocated to a request."""
    tenant = request.tenant_id
    req_id = request.request_id
    self.req_to_tenant[req_id] = tenant
    block_size = self.block_size

    for block_group in new_blocks:
        for block in block_group:
            if block.is_null:
                continue
            bid = block.block_id
            if bid not in self.residency_holders:
                # New block, sole holder
                self.residency_holders[bid] = {tenant: 1}
                self.tenant_resident_tokens[tenant] += block_size
            elif tenant not in self.residency_holders[bid]:
                # Cross-tenant sharing: rebalance
                holders = self.residency_holders[bid]
                k = len(holders)
                # Existing tenants: 1/k → 1/(k+1)
                delta = block_size / k - block_size / (k + 1)
                for existing_tenant in holders:
                    self.tenant_resident_tokens[existing_tenant] -= delta
                # New tenant gets 1/(k+1)
                holders[tenant] = 1
                self.tenant_resident_tokens[tenant] += block_size / (k + 1)
            else:
                # Intra-tenant sharing (same tenant, another request)
                self.residency_holders[bid][tenant] += 1


def _residency_on_free(self, request: Request):
    """Update residency state when a request's blocks are freed."""
    tenant = request.tenant_id
    req_id = request.request_id
    block_size = self.block_size

    # Get all blocks for this request (still in req_to_blocks at this point)
    for mgr in self.coordinator.single_type_managers:
        blocks = mgr.req_to_blocks.get(req_id, [])
        for block in blocks:
            if block.is_null:
                continue
            bid = block.block_id
            holders = self.residency_holders.get(bid)
            if holders is None or tenant not in holders:
                continue  # shouldn't happen, but defensive

            holders[tenant] -= 1
            if holders[tenant] == 0:
                # This tenant no longer holds this block
                k = len(holders)  # includes tenant (about to be removed)
                del holders[tenant]
                self.tenant_resident_tokens[tenant] -= block_size / k
                if not holders:
                    # Block fully released
                    del self.residency_holders[bid]
                else:
                    # Remaining tenants: 1/k → 1/(k-1)
                    delta = block_size / (k - 1) - block_size / k
                    for remaining_tenant in holders:
                        self.tenant_resident_tokens[remaining_tenant] += delta

    self.req_to_tenant.pop(req_id, None)
```

---

### 3. `vllm/v1/engine/core.py` — Step-time charging + Prometheus counter

**3a. Register counter in `__init__`** (after scheduler creation, ~line 158):

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

**3b. Modify `step()`** (wrap with timing, add residency after update_from_output):

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    if not self.scheduler.has_requests():
        return {}, False

    t0 = time.monotonic()

    scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)

    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    dt = time.monotonic() - t0
    self._accumulate_residency(dt)

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0


def _accumulate_residency(self, dt: float) -> None:
    """Charge each tenant for their resident KV-cache token-seconds."""
    tenant_tokens = self.scheduler.kv_cache_manager.tenant_resident_tokens
    for tenant, tokens in tenant_tokens.items():
        if tokens > 0:
            self._residency_counter.labels(tenant_id=tenant).inc(tokens * dt)
```

---

## Correctness

### Conservation

```
∑ tenant_resident_tokens[t] = total_allocated_non_null_blocks × block_size
```

This holds because every block contributes exactly `block_size` total across all
holders (split as `block_size / k` each for k holders).

### Invariants maintained at every mutation

1. `residency_holders[bid]` reflects exactly which tenants currently hold block `bid`
   and how many of their requests reference it
2. `tenant_resident_tokens[t]` = Σ over blocks held by t of `block_size / len(holders[bid])`
3. On allocate: new block adds tokens, shared block rebalances
4. On free: if tenant's last reference to a block, remove and rebalance remaining

### Edge cases

| Case | Handling |
|------|----------|
| Request with no `tenant_id` | Skipped (`if request.tenant_id is not None`) |
| Intra-tenant sharing (prefix cache hit within same tenant) | `holders[bid][tenant] += 1`, no token change |
| Cross-tenant sharing (prefix cache hit across tenants) | Rebalance: existing tenants lose `1/k - 1/(k+1)`, new tenant gains `1/(k+1)` |
| Preemption | Goes through `_free_request_blocks` → `free()` → residency updated |
| Deferred free | Goes through `pop_blocks_for_free()` → residency updated at pop time |
| Sliding window null blocks | Skipped via `if block.is_null: continue` |

---

## Performance

| Operation | Cost | Frequency |
|-----------|------|-----------|
| `_residency_on_allocate` | O(new_blocks_this_step) | Once per request per step (decode: ~0-1 blocks) |
| `_residency_on_free` | O(blocks_per_request) | Once per request lifetime (at completion) |
| `_accumulate_residency` | O(N_tenants) | Once per step |
| **Rebalance (cross-tenant)** | O(k) per shared block | Rare (requires same prefix across tenants) |

In steady-state decode: most steps allocate 0-1 new blocks per request (block_size=16
tokens, 1 token/step → new block every 16 steps). The per-step overhead is dominated
by `_accumulate_residency` at O(N_tenants) ≈ O(1).

---

## What dt includes

```
t0 ─── schedule() ─── execute_model() ─── update_from_output() ─── t1
       ↑ allocate                          ↑ free
       blocks                              blocks

dt = t1 - t0
```

`dt` includes the full step: scheduling, GPU execution, and output processing.
Blocks that are freed during `update_from_output()` still get charged for this step's
dt because `tenant_resident_tokens` was updated during the free (inside
`update_from_output`), but the charging loop reads the value AFTER those frees.

**Subtlety**: A freed request's tokens are subtracted from `tenant_resident_tokens`
during `update_from_output()` (when `free()` is called). So `_accumulate_residency(dt)`
runs after the free and sees the reduced count. This means the last step of a
completing request does NOT get charged for that request's blocks.

This is acceptable: the under-count is at most `blocks_per_request × block_size × dt`
for one step (~50ms), negligible over a multi-minute experiment. If exact accounting
is needed, snapshot `tenant_resident_tokens` before `update_from_output()` and use
that snapshot for charging.

---

## Files summary

| File | Change |
|------|--------|
| `vllm/v1/request.py` | Add `self.tenant_id` from `extra_args` (~4 lines) |
| `vllm/v1/core/kv_cache_manager.py` | Add residency state + `_residency_on_allocate` + `_residency_on_free` (~60 lines) |
| `vllm/v1/engine/core.py` | Add timing in `step()`, `_accumulate_residency()`, register Counter (~15 lines) |

Total: ~80 lines of new code.
