#!/usr/bin/env python3
"""
Workload driver for vLLM residency experiments.

Sends symmetric Poisson-arrival load from N tenants, measures per-request
latency metrics (TTFT, ITL, E2E) via streaming SSE, scrapes residency
Prometheus counter, and outputs JSON summary + per-request CSV.
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass, field

import aiohttp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    tenant_id: str
    request_idx: int
    ttft_ms: float
    itl_ms_list: list = field(default_factory=list)
    e2e_ms: float = 0.0
    num_output_tokens: int = 0
    start_time: float = 0.0


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

# Small vocabulary for generating random prompts. We use words rather than
# random characters so the tokenizer produces roughly 1 token per word.
_VOCAB = [
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
    "was", "on", "are", "as", "with", "his", "they", "at", "be", "this",
    "from", "or", "had", "by", "not", "but", "some", "what", "there",
    "we", "can", "out", "other", "were", "all", "your", "when", "up",
    "use", "how", "each", "she", "which", "do", "their", "time", "if",
    "will", "way", "about", "many", "then", "them", "would", "write",
    "like", "so", "these", "her", "long", "make", "thing", "see", "him",
    "two", "has", "look", "more", "day", "could", "go", "come", "did",
    "my", "no", "most", "who", "over", "know", "water", "than", "call",
    "first", "people", "may", "down", "side", "been", "now", "find",
    "head", "stand", "own", "page", "should", "country", "found", "answer",
    "school", "grow", "study", "still", "learn", "plant", "cover", "food",
    "sun", "four", "between", "state", "keep", "eye", "never", "last",
    "let", "thought", "city", "tree", "cross", "farm", "hard", "start",
    "might", "story", "saw", "far", "sea", "draw", "left", "late", "run",
]


def generate_prompt(num_tokens: int, rng: random.Random) -> str:
    """Generate a random prompt of approximately `num_tokens` tokens."""
    # Approximate: 1 word ≈ 1 token (conservative; pads slightly)
    words = [rng.choice(_VOCAB) for _ in range(num_tokens)]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Streaming request + metrics
# ---------------------------------------------------------------------------

async def generate_request(
    session: aiohttp.ClientSession,
    tenant_id: str,
    request_idx: int,
    config: argparse.Namespace,
    rng: random.Random,
) -> RequestResult:
    """Send one streaming chat completion request and measure TTFT/ITL/E2E."""

    prompt_text = generate_prompt(config.prompt_tokens, rng)

    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": config.max_tokens,
        "stream": True,
        "user": tenant_id,  # vLLM uses this as tenant identifier
    }

    headers = {"Content-Type": "application/json"}

    result = RequestResult(
        tenant_id=tenant_id,
        request_idx=request_idx,
        start_time=time.time(),
    )

    t_start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0

    try:
        async with session.post(
            f"{config.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"HTTP {resp.status} from server: {body[:200]}"
                )

            # Parse SSE stream
            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Check if this chunk contains a token
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if not content:
                    continue

                now = time.perf_counter()
                token_count += 1

                if first_token_time is None:
                    first_token_time = now
                else:
                    # Record inter-token latency
                    itl = (now - last_token_time) * 1000.0  # ms
                    result.itl_ms_list.append(itl)

                last_token_time = now

    except Exception as e:
        # On error, record what we have with sentinel values
        result.ttft_ms = -1.0
        result.e2e_ms = -1.0
        result.num_output_tokens = token_count
        return result

    t_end = time.perf_counter()

    result.e2e_ms = (t_end - t_start) * 1000.0
    result.ttft_ms = (
        (first_token_time - t_start) * 1000.0
        if first_token_time is not None
        else -1.0
    )
    result.num_output_tokens = token_count

    return result


# ---------------------------------------------------------------------------
# Per-tenant Poisson worker
# ---------------------------------------------------------------------------

async def tenant_worker(
    tenant_id: str,
    config: argparse.Namespace,
    results: list,
    seed: int,
):
    """Generate Poisson-arrival requests for one tenant until duration expires."""

    rng = random.Random(seed)
    rate = config.rate
    deadline = time.time() + config.duration
    request_idx = 0
    tasks = []

    connector = aiohttp.TCPConnector(limit=0)  # no connection limit
    timeout = aiohttp.ClientTimeout(total=config.duration + 120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while time.time() < deadline:
            # Poisson inter-arrival time
            wait = rng.expovariate(rate)
            await asyncio.sleep(wait)

            if time.time() >= deadline:
                break

            # Fire request (don't await — let it run concurrently)
            task = asyncio.create_task(
                generate_request(session, tenant_id, request_idx, config, rng)
            )
            tasks.append(task)
            request_idx += 1

        # Wait for all in-flight requests to finish (with timeout)
        if tasks:
            done, pending = await asyncio.wait(
                tasks, timeout=120, return_when=asyncio.ALL_COMPLETED
            )
            # Cancel any still-pending requests
            for t in pending:
                t.cancel()
            # Collect results
            for t in done:
                try:
                    result = t.result()
                    results.append(result)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Prometheus scraping
# ---------------------------------------------------------------------------

async def scrape_residency(base_url: str, tenants: list) -> dict:
    """Scrape /metrics for residency_token_seconds_total per tenant."""

    residency = {}
    pattern = re.compile(
        r'vllm:residency_token_seconds_total\{[^}]*tenant_id="([^"]+)"[^}]*\}\s+([\d.e+\-]+)'
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/metrics") as resp:
                if resp.status != 200:
                    return {t: 0.0 for t in tenants}
                text = await resp.text()

        for match in pattern.finditer(text):
            tid = match.group(1)
            value = float(match.group(2))
            if tid in tenants:
                residency[tid] = value

    except Exception:
        pass

    # Fill missing tenants with 0
    for t in tenants:
        if t not in residency:
            residency[t] = 0.0

    return residency


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(data: list, p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    d = k - f
    return data[f] * (1 - d) + data[c] * d


def compute_stats(values: list) -> dict:
    """Compute mean, p50, p95, p99 for a list of values."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_vals = sorted(values)
    mean_val = sum(sorted_vals) / len(sorted_vals)
    return {
        "mean": round(mean_val, 2),
        "p50": round(percentile(sorted_vals, 50), 2),
        "p95": round(percentile(sorted_vals, 95), 2),
        "p99": round(percentile(sorted_vals, 99), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Workload driver for vLLM residency experiments"
    )
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="vLLM server URL")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                        help="Model name")
    parser.add_argument("--tenants", default="tenant_A,tenant_B,tenant_C",
                        help="Comma-separated tenant IDs")
    parser.add_argument("--prompt-tokens", type=int, default=1024,
                        help="Approximate prompt length in tokens")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max output tokens per request")
    parser.add_argument("--rate", type=float, default=2.0,
                        help="Requests/sec per tenant (Poisson rate)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Experiment duration in seconds")
    parser.add_argument("--output-dir", default="./results",
                        help="Directory for output files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    config = parser.parse_args()
    tenants = [t.strip() for t in config.tenants.split(",")]

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Starting workload driver")
    print(f"  Server:   {config.base_url}")
    print(f"  Model:    {config.model}")
    print(f"  Tenants:  {tenants}")
    print(f"  Rate:     {config.rate} req/s per tenant")
    print(f"  Duration: {config.duration}s")
    print(f"  Prompt:   ~{config.prompt_tokens} tokens")
    print(f"  Max out:  {config.max_tokens} tokens")
    print()

    # Collect all results across tenants
    all_results: list[RequestResult] = []

    # Launch one worker per tenant concurrently
    t_experiment_start = time.time()

    tasks = []
    for i, tenant_id in enumerate(tenants):
        tenant_results: list[RequestResult] = []
        all_results.append(tenant_results)  # type: ignore
        task = asyncio.create_task(
            tenant_worker(tenant_id, config, tenant_results, config.seed + i)
        )
        tasks.append(task)

    await asyncio.gather(*tasks)

    t_experiment_end = time.time()
    total_duration = t_experiment_end - t_experiment_start

    # Flatten results
    flat_results: list[RequestResult] = []
    for tenant_results in all_results:
        flat_results.extend(tenant_results)  # type: ignore

    print(f"\nExperiment complete: {len(flat_results)} requests in {total_duration:.1f}s")

    # Scrape residency metrics
    print("Scraping residency metrics...")
    residency = await scrape_residency(config.base_url, tenants)

    # ---------------------------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------------------------
    csv_path = os.path.join(config.output_dir, "requests.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tenant_id", "request_idx", "ttft_ms", "mean_itl_ms",
            "p50_itl_ms", "p95_itl_ms", "p99_itl_ms", "e2e_ms",
            "num_output_tokens", "start_time",
        ])
        for r in flat_results:
            itl_stats = compute_stats(r.itl_ms_list)
            writer.writerow([
                r.tenant_id,
                r.request_idx,
                round(r.ttft_ms, 2),
                itl_stats["mean"],
                itl_stats["p50"],
                itl_stats["p95"],
                itl_stats["p99"],
                round(r.e2e_ms, 2),
                r.num_output_tokens,
                round(r.start_time, 3),
            ])

    print(f"  Wrote {csv_path} ({len(flat_results)} rows)")

    # ---------------------------------------------------------------------------
    # Build summary
    # ---------------------------------------------------------------------------
    per_tenant = {}
    for tenant_id in tenants:
        tenant_reqs = [r for r in flat_results if r.tenant_id == tenant_id]
        # Filter out failed requests for stats
        valid_reqs = [r for r in tenant_reqs if r.ttft_ms >= 0]

        ttft_values = [r.ttft_ms for r in valid_reqs]
        e2e_values = [r.e2e_ms for r in valid_reqs]
        # Flatten all ITL values across requests for this tenant
        all_itl = []
        for r in valid_reqs:
            all_itl.extend(r.itl_ms_list)

        total_tokens = sum(r.num_output_tokens for r in tenant_reqs)

        per_tenant[tenant_id] = {
            "num_requests": len(tenant_reqs),
            "num_successful": len(valid_reqs),
            "ttft_ms": compute_stats(ttft_values),
            "itl_ms": compute_stats(all_itl),
            "e2e_ms": compute_stats(e2e_values),
            "total_output_tokens": total_tokens,
            "residency_token_seconds": residency.get(tenant_id, 0.0),
        }

    summary = {
        "config": {
            "base_url": config.base_url,
            "model": config.model,
            "tenants": tenants,
            "prompt_tokens": config.prompt_tokens,
            "max_tokens": config.max_tokens,
            "rate": config.rate,
            "duration": config.duration,
            "seed": config.seed,
        },
        "per_tenant": per_tenant,
        "totals": {
            "total_requests": len(flat_results),
            "total_duration_s": round(total_duration, 2),
            "residency_sum": round(sum(residency.values()), 2),
        },
    }

    json_path = os.path.join(config.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Wrote {json_path}")

    # Print quick summary
    print("\n--- Summary ---")
    for tid in tenants:
        s = per_tenant[tid]
        print(
            f"  {tid}: {s['num_requests']} reqs, "
            f"TTFT p50={s['ttft_ms']['p50']:.0f}ms, "
            f"E2E p50={s['e2e_ms']['p50']:.0f}ms, "
            f"residency={s['residency_token_seconds']:.1f}"
        )
    print(f"  Total: {len(flat_results)} requests, {total_duration:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
