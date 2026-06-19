# vLLM v0.23.0 Patch Specification: Per-Tenant KV-Cache Residency

## Overview

This spec describes **exactly** how to patch 3 files in vLLM v0.23.0
(`vllm/vllm-openai:v0.23.0` Docker image, commit `0fc695fc6`) to add a
Prometheus counter that accumulates per-tenant KV-cache residency in
token-seconds.

**Target base**: vLLM v0.23.0 (tag `v0.23.0` on `github.com/vllm-project/vllm`)

**Output**: 3 patched Python files placed in `vllm/v1/` in this repo, overlaid
onto the base image via `cp -r` (see Dockerfile).

---

## Instructions

For each file below:
1. Clone vLLM at v0.23.0: `git clone --depth 1 --branch v0.23.0 git@github.com:vllm-project/vllm.git`
2. Copy the original file into this repo at the path shown
3. Apply the edits described using the **exact anchor text** provided
4. The anchor text is the literal string in the original file to search for

---

## File 1: `vllm/v1/request.py`

**Copy from**: `<vllm-clone>/vllm/v1/request.py`
**Place at**: `<this-repo>/vllm/v1/request.py`

### Edit 1a: Add import (no edit needed — `time` is already imported at line 5)

No new imports required for this file.

### Edit 1b: Add `tenant_id` in the `pooling_params` branch

**Find this exact text** (around line 104–106):
```python
        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
```

**Replace with**:
```python
        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
            self.tenant_id: str | None = None
```

### Edit 1c: Add `tenant_id` extraction from `extra_args`

**Find this exact text** (around line 114–117):
```python
            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
```

**Replace with**:
```python
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

**IMPORTANT**: The original code after line 117 is:
```python
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")
```
That outer `else` (for the `elif sampling_params is not None` branch) must remain
unchanged. The new inner `else` (for `if sampling_params.extra_args is not None`)
goes BEFORE the outer else. The full resulting block should read:

```python
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
                self.tenant_id: str | None = sampling_params.extra_args.get(
                    "tenant_id"
                )
            else:
                self.tenant_id: str | None = None
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")
```

---

## File 2: `vllm/v1/core/kv_cache_manager.py`

**Copy from**: `<vllm-clone>/vllm/v1/core/kv_cache_manager.py`
**Place at**: `<this-repo>/vllm/v1/core/kv_cache_manager.py`

### Edit 2a: Add `defaultdict` import

**Find this exact text** (line 1–7):
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload
```

**Replace with**:
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload
```

### Edit 2b: Add residency state in `__init__`

**Find this exact text** (around line 166–173):
```python
        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )
```

**Replace with**:
```python
        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # --- Residency instrumentation ---
        # block_id → {tenant_id: request_refcount}
        self.residency_holders: dict[int, dict[str, int]] = {}
        # tenant_id → current fractional resident token count
        self.tenant_resident_tokens: dict[str, float] = defaultdict(float)
        # request_id → tenant_id (for free-time lookup)
        self.req_to_tenant: dict[str, str] = {}
        # block_size used for residency accounting
        self._residency_block_size: int = (
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            if kv_cache_config.kv_cache_groups
            else 16
        )
```

### Edit 2c: Hook residency tracking into `allocate_slots()`

The `allocate_slots` method has the following structure (simplified):
```
line 238: def allocate_slots(self, request, ...):
line 331-398: ... validation, checks, may return None ...
line 400-411: allocate_new_computed_blocks (prefix cache hits)
line 413: new_blocks = self.coordinator.allocate_new_blocks(...)
line 422-423: early return (no caching / delay_cache_blocks)
line 430-434: caching logic
line 436: return self.create_kv_cache_blocks(new_blocks)
```

There are TWO return paths. The residency hook must run AFTER `allocate_new_blocks`
but BEFORE both returns.

**Find this exact text** (around line 413–436):
```python
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)
```

**Replace with**:
```python
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # --- Residency: track all newly allocated blocks ---
        if getattr(request, 'tenant_id', None) is not None:
            self._residency_on_allocate(request, new_blocks)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)
```

### Edit 2d: Hook residency tracking into prefix-cache allocation

When prefix-cached blocks are reused, they go through `allocate_new_computed_blocks`
(NOT through `allocate_new_blocks`). We must track these too.

**Find this exact text** (around line 400–411):
```python
        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )
```

**Replace with**:
```python
        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )
            # --- Residency: track prefix-cached blocks assigned to request ---
            if getattr(request, 'tenant_id', None) is not None:
                self._residency_on_allocate(request, new_computed_block_list)
```

### Edit 2e: Hook residency tracking into `free()`

**Find this exact text** (around line 438–446):
```python
    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.coordinator.free(request.request_id)
```

**Replace with**:
```python
    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        # --- Residency: update BEFORE blocks are freed (req_to_blocks still populated) ---
        if getattr(request, 'tenant_id', None) is not None:
            self._residency_on_free(request)

        self.coordinator.free(request.request_id)
```

### Edit 2f: Add the two new methods

**Insert the following two methods at the end of the `KVCacheManager` class**
(after the last method in the class, before any other class definition or EOF).

Find the last method of `KVCacheManager`. In v0.23.0, the class ends around line 520
(after `reset_prefix_cache` and `get_computed_blocks`). Add after the last method:

```python
    # ─── Residency instrumentation methods ───────────────────────────

    def _residency_on_allocate(
        self,
        request: Request,
        new_blocks: tuple[Sequence[KVCacheBlock], ...],
    ) -> None:
        """Update residency state when blocks are allocated to a request.

        Called for both freshly-allocated blocks (from allocate_new_blocks)
        and prefix-cached blocks (from allocate_new_computed_blocks).
        """
        tenant = request.tenant_id
        self.req_to_tenant[request.request_id] = tenant
        block_size = self._residency_block_size

        for block_group in new_blocks:
            for block in block_group:
                if block.is_null:
                    continue
                bid = block.block_id
                if bid not in self.residency_holders:
                    # Fresh block, sole holder
                    self.residency_holders[bid] = {tenant: 1}
                    self.tenant_resident_tokens[tenant] += block_size
                elif tenant not in self.residency_holders[bid]:
                    # Cross-tenant sharing: rebalance all holders
                    holders = self.residency_holders[bid]
                    k = len(holders)
                    # Existing tenants go from block_size/k to block_size/(k+1)
                    delta = block_size / k - block_size / (k + 1)
                    for existing_tenant in holders:
                        self.tenant_resident_tokens[existing_tenant] -= delta
                    # New tenant gets block_size/(k+1)
                    holders[tenant] = 1
                    self.tenant_resident_tokens[tenant] += block_size / (k + 1)
                else:
                    # Intra-tenant sharing (same tenant, different request
                    # reusing a prefix-cached block). No token change —
                    # tenant was already credited.
                    self.residency_holders[bid][tenant] += 1

    def _residency_on_free(self, request: Request) -> None:
        """Update residency state when a request's blocks are about to be freed.

        Must be called BEFORE coordinator.free() since that pops req_to_blocks.
        """
        tenant = request.tenant_id
        req_id = request.request_id
        block_size = self._residency_block_size

        # Iterate all blocks currently held by this request.
        # coordinator.single_type_managers[*].req_to_blocks has the mapping.
        for mgr in self.coordinator.single_type_managers:
            blocks = mgr.req_to_blocks.get(req_id, [])
            for block in blocks:
                if block.is_null:
                    continue
                bid = block.block_id
                holders = self.residency_holders.get(bid)
                if holders is None or tenant not in holders:
                    # Shouldn't happen, but defensive
                    continue

                holders[tenant] -= 1
                if holders[tenant] == 0:
                    # Tenant's last reference to this block
                    k = len(holders)  # includes tenant (about to remove)
                    del holders[tenant]
                    self.tenant_resident_tokens[tenant] -= block_size / k

                    if not holders:
                        # Block fully released — no one holds it
                        del self.residency_holders[bid]
                    else:
                        # Rebalance remaining: 1/k → 1/(k-1)
                        delta = block_size / (k - 1) - block_size / k
                        for remaining_tenant in holders:
                            self.tenant_resident_tokens[remaining_tenant] += delta

        self.req_to_tenant.pop(req_id, None)
```

---

## File 3: `vllm/v1/engine/core.py`

**Copy from**: `<vllm-clone>/vllm/v1/engine/core.py`
**Place at**: `<this-repo>/vllm/v1/engine/core.py`

### Edit 3a: Add `Counter` import

**Find this exact text** (near top of file, around line 8):
```python
import time
```

This confirms `time` is already imported. Now add the prometheus import.

**Find this exact text** (around line 21):
```python
import zmq
```

**Replace with**:
```python
import zmq
from prometheus_client import Counter
```

### Edit 3b: Register the Prometheus counter in `EngineCore.__init__`

**Find this exact text** (around line 157–160):
```python
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore
```

**Replace with**:
```python
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        # --- Residency instrumentation: Prometheus counter ---
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

### Edit 3c: Add timing and residency charging to `step()`

**Find this exact text** (around line 443–472):
```python
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**Replace with**:
```python
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False

        t0 = time.monotonic()

        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        dt = time.monotonic() - t0
        self._accumulate_residency(dt)

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def _accumulate_residency(self, dt: float) -> None:
        """Charge each tenant for their resident KV-cache token-seconds.

        Reads tenant_resident_tokens from the KVCacheManager (maintained
        incrementally at allocate/free time) and increments the Prometheus
        counter by tokens * dt for each tenant.
        """
        tenant_tokens = self.scheduler.kv_cache_manager.tenant_resident_tokens
        for tenant, tokens in tenant_tokens.items():
            if tokens > 0:
                self._residency_counter.labels(tenant_id=tenant).inc(tokens * dt)
```

---

## Verification

After applying all edits, verify:

### 1. Syntax check
```bash
python3 -c "import ast; ast.parse(open('vllm/v1/request.py').read())"
python3 -c "import ast; ast.parse(open('vllm/v1/core/kv_cache_manager.py').read())"
python3 -c "import ast; ast.parse(open('vllm/v1/engine/core.py').read())"
```

### 2. Docker build
```bash
docker build -t residency-vllm:test .
```

### 3. Quick smoke test (requires GPU)
```bash
docker run --gpus all -p 8000:8000 residency-vllm:test \
    --model meta-llama/Llama-3.1-8B --max-model-len 2048

# In another terminal:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 16,
    "extra_body": {"vllm_xargs": {"tenant_id": "test_tenant"}}
  }'

# Check metric appeared:
curl -s http://localhost:8000/metrics | grep residency_token_seconds
# Expected: vllm:residency_token_seconds_total{tenant_id="test_tenant"} <some_value>
```

### 4. Conservation check
```bash
curl -s http://localhost:8000/metrics | grep residency_token_seconds
# Sum of all tenant values should approximate: num_allocated_blocks × block_size × elapsed_time
```

---

## Key facts for implementers

| Item | Value |
|------|-------|
| Base image | `vllm/vllm-openai:v0.23.0` |
| Base commit | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| `time` already imported in `core.py`? | Yes (line 8) |
| `prometheus_client` in base image? | Yes (pre-installed) |
| `defaultdict` needs import in `kv_cache_manager.py`? | Yes (not already imported) |
| `KVCacheBlock.block_id` attribute | Exists (int, 0 to num_gpu_blocks-1) |
| `KVCacheBlock.is_null` attribute | Exists (bool, marks sentinel blocks) |
| `coordinator.single_type_managers` | List of `SingleTypeKVCacheManager` instances |
| `mgr.req_to_blocks` | `defaultdict[str, list[KVCacheBlock]]` — maps request_id → blocks |
| `self.scheduler.kv_cache_manager` | Direct attribute on `Scheduler` (line 231 in scheduler.py) |
| `schedule()` signature at v0.23.0 | `def schedule(self)` — takes NO arguments |
| `pop_blocks_for_free` exists? | NO — does not exist in v0.23.0 |
| Thread safety | Not needed — scheduler runs single-threaded in vLLM v1 |

---

## Design decisions

### Why `getattr(request, 'tenant_id', None)` instead of `request.tenant_id`

Defensive: if a request somehow bypasses the patched `Request.__init__` (e.g.,
created by internal vLLM code without the field), the `getattr` avoids an
`AttributeError` crash. In normal operation, `tenant_id` is always set.

### Why hook both `allocate_new_blocks` AND `allocate_new_computed_blocks`

- `allocate_new_blocks`: allocates fresh physical blocks for new tokens
- `allocate_new_computed_blocks`: attaches existing prefix-cached blocks to a request

Both give a request ownership of blocks. If we only hook the former, prefix-cached
blocks used by a request wouldn't be counted in that tenant's residency — violating
conservation. In the symmetric-workload experiment (no prefix hits), only the first
hook fires. But the second hook ensures correctness in general.

### Why `_residency_on_free` must run BEFORE `coordinator.free()`

`coordinator.free()` calls `mgr.req_to_blocks.pop(request_id)`, which removes
the block list. Our hook needs that list to know which blocks to decrement.
So we must run first.

### Conservation invariant

At any moment:
```
sum(tenant_resident_tokens.values()) == count_of_allocated_non_null_blocks × block_size
```

Every physical block contributes exactly `block_size` total across all its holders
(split as `block_size / k` for k holders). Allocate adds, free removes, rebalance
redistributes — the sum is always conserved.

### Last-step under-count

Requests freed during `update_from_output()` have their `tenant_resident_tokens`
decremented BEFORE `_accumulate_residency(dt)` runs. So the last step's residency
for completing requests goes uncharged. This is at most `blocks × block_size × dt`
(~50ms worth) per request — negligible over a multi-minute experiment.
