#!/usr/bin/env python3
"""
Post-process blis observe trace output into summary.json + requests.csv.

Reads:
  - trace.csv        (per-request metrics from blis observe)
  - trace.itl.csv    (optional; per-chunk ITL timestamps)

Produces:
  - summary.json     (same schema generate_figures.py expects)
  - requests.csv     (same schema as workload_driver.py output)

Also scrapes /metrics from the vLLM server for residency counters (patched
variant only; gracefully returns 0 if unavailable).
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
import urllib.error
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(values):
    """Compute summary statistics matching generate_figures.py expectations.

    Returns dict with: mean, min, max, p50, median, p90, p99.
    (p50 and median are identical — both included for compatibility with
    generate_figures.py which checks p50 first, then falls back to median.)
    """
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "median": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
    }


# ---------------------------------------------------------------------------
# Residency scraping (same logic as workload_driver.py)
# ---------------------------------------------------------------------------

def scrape_residency(server_url, tenants):
    """Scrape /metrics for vllm:residency_token_seconds_total per tenant."""
    residency = {t: 0.0 for t in tenants}
    pattern = re.compile(
        r'vllm:residency_token_seconds_total\{[^}]*tenant_id="([^"]+)"[^}]*\}\s+([\d.e+\-]+)'
    )
    try:
        url = f"{server_url}/metrics"
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8")
        for match in pattern.finditer(text):
            tid = match.group(1)
            value = float(match.group(2))
            if tid in residency:
                residency[tid] = value
    except (urllib.error.URLError, OSError):
        pass
    return residency


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

def parse_trace_csv(path):
    """Parse blis observe trace.csv into a list of request dicts."""
    requests = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip failed requests
            status = row.get("status", "").strip()
            if status != "ok" and status != "":
                # Allow empty status (some versions omit it for success)
                if status not in ("ok", "success", ""):
                    continue

            req = {
                "request_id": row.get("request_id", ""),
                "client_id": row.get("client_id", ""),
                "tenant_id": row.get("tenant_id", row.get("client_id", "")),
                "input_tokens": int(row.get("input_tokens", 0) or 0),
                "output_tokens": int(row.get("output_tokens", 0) or 0),
                "server_input_tokens": int(row.get("server_input_tokens", 0) or 0),
                "send_time_us": int(row.get("send_time_us", 0) or 0),
                "first_chunk_time_us": int(row.get("first_chunk_time_us", 0) or 0),
                "last_chunk_time_us": int(row.get("last_chunk_time_us", 0) or 0),
                "num_chunks": int(row.get("num_chunks", 0) or 0),
                "status": status,
            }
            requests.append(req)
    return requests


def parse_itl_csv(path):
    """Parse blis observe trace.itl.csv into per-request ITL lists.

    Returns dict: request_id -> list of ITL values in ms.
    """
    itl_by_request = defaultdict(list)
    if not os.path.isfile(path):
        return itl_by_request

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("request_id", "")
            # ITL file has per-chunk timestamps; compute deltas
            chunk_time_us = int(row.get("timestamp_us", row.get("chunk_time_us", 0)) or 0)
            itl_by_request[rid].append(chunk_time_us)

    # Convert absolute timestamps to inter-chunk deltas (ms)
    result = {}
    for rid, timestamps in itl_by_request.items():
        if len(timestamps) < 2:
            result[rid] = []
            continue
        timestamps.sort()
        deltas = [(timestamps[i] - timestamps[i - 1]) / 1000.0
                  for i in range(1, len(timestamps))]
        result[rid] = deltas
    return result


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_request_metrics(req, itl_map):
    """Compute per-request latency metrics from trace row."""
    send = req["send_time_us"]
    first_chunk = req["first_chunk_time_us"]
    last_chunk = req["last_chunk_time_us"]

    ttft_ms = (first_chunk - send) / 1000.0 if first_chunk > send else 0.0
    e2e_ms = (last_chunk - send) / 1000.0 if last_chunk > send else 0.0

    # ITL from itl_map if available
    itl_list = itl_map.get(req["request_id"], [])

    # Mean ITL
    mean_itl_ms = float(np.mean(itl_list)) if itl_list else 0.0

    return {
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "itl_list": itl_list,
        "mean_itl_ms": mean_itl_ms,
        "num_output_tokens": req["output_tokens"],
        "num_input_tokens": req["input_tokens"],
    }


def build_group_summary(metrics_list, itl_lists):
    """Build stats dict for a group of requests (flat format for generate_figures.py)."""
    ttft_values = [m["ttft_ms"] for m in metrics_list if m["ttft_ms"] > 0]
    e2e_values = [m["e2e_ms"] for m in metrics_list if m["e2e_ms"] > 0]
    all_itl = [v for itl in itl_lists for v in itl]

    total_input = sum(m["num_input_tokens"] for m in metrics_list)
    total_output = sum(m["num_output_tokens"] for m in metrics_list)

    # Throughput: use e2e span
    if e2e_values and len(metrics_list) > 0:
        total_time_s = max(e2e_values) / 1000.0 if e2e_values else 1.0
    else:
        total_time_s = 1.0

    return {
        "num_requests": len(metrics_list),
        "num_successful": len(ttft_values),
        "ttft_ms": compute_stats(ttft_values),
        "itl_ms": compute_stats(all_itl),
        "e2e_ms": compute_stats(e2e_values),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "throughput": {
            "input_tokens_per_sec": round(total_input / total_time_s, 2) if total_time_s > 0 else 0.0,
            "output_tokens_per_sec": round(total_output / total_time_s, 2) if total_time_s > 0 else 0.0,
            "requests_per_sec": round(len(metrics_list) / total_time_s, 2) if total_time_s > 0 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-csv", required=True,
                        help="Path to blis observe trace.csv")
    parser.add_argument("--itl-csv", default=None,
                        help="Path to blis observe trace.itl.csv (optional)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write summary.json and requests.csv")
    parser.add_argument("--server-url", default="http://localhost:8000",
                        help="vLLM server URL for residency scraping")
    parser.add_argument("--model", default="Qwen/Qwen3-14B",
                        help="Model name for config metadata")
    parser.add_argument("--rate", type=float, default=2.0,
                        help="Per-tenant request rate for config metadata")
    parser.add_argument("--duration", type=int, default=300,
                        help="Experiment duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse trace data
    print(f"Parsing trace: {args.trace_csv}")
    requests = parse_trace_csv(args.trace_csv)
    if not requests:
        print("ERROR: No requests found in trace.csv", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(requests)} requests parsed")

    # Parse ITL data if available
    itl_path = args.itl_csv
    if itl_path is None:
        # Default: same directory as trace.csv, named trace.itl.csv
        itl_path = os.path.join(os.path.dirname(args.trace_csv), "trace.itl.csv")
    itl_map = parse_itl_csv(itl_path)
    if itl_map:
        print(f"  ITL data loaded for {len(itl_map)} requests")

    # Discover tenants
    tenants = sorted(set(r["tenant_id"] for r in requests))
    print(f"  Tenants: {tenants}")

    # Compute per-request metrics
    all_metrics = []
    all_itl_lists = []
    tenant_metrics = defaultdict(list)
    tenant_itl_lists = defaultdict(list)

    for req in requests:
        m = compute_request_metrics(req, itl_map)
        all_metrics.append(m)
        all_itl_lists.append(m["itl_list"])
        tenant_metrics[req["tenant_id"]].append(m)
        tenant_itl_lists[req["tenant_id"]].append(m["itl_list"])

    # Scrape residency metrics
    print("Scraping residency metrics...")
    residency = scrape_residency(args.server_url, tenants)

    # Build per-tenant summary (FLAT format for generate_figures.py)
    per_tenant = {}
    for tid in tenants:
        t_metrics = tenant_metrics[tid]
        t_itl = tenant_itl_lists[tid]
        summary = build_group_summary(t_metrics, t_itl)
        summary["residency_token_seconds"] = residency.get(tid, 0.0)
        per_tenant[tid] = summary

    # Build overall summary
    overall = build_group_summary(all_metrics, all_itl_lists)
    overall["residency_token_seconds_total"] = round(sum(residency.values()), 2)
    # Also nest latency under "latency" key for run_ab_experiment.sh compatibility
    overall["latency"] = {
        "ttft_ms": overall["ttft_ms"],
        "itl_ms": overall["itl_ms"],
        "e2e_ms": overall["e2e_ms"],
    }

    # Assemble final summary.json
    aggregate_rate = args.rate * len(tenants)
    summary_doc = {
        "config": {
            "model": args.model,
            "tenants": tenants,
            "prompt_tokens": 1024,
            "max_tokens": 128,
            "rate_per_tenant": args.rate,
            "rate": args.rate,
            "aggregate_rate": aggregate_rate,
            "duration": args.duration,
            "seed": args.seed,
        },
        "overall": overall,
        "per_tenant": per_tenant,
        "totals": {
            "total_requests": len(requests),
            "total_successful": sum(1 for m in all_metrics if m["ttft_ms"] > 0),
            "total_duration_s": args.duration,
        },
    }

    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_doc, f, indent=2)
    print(f"  Wrote {json_path}")

    # Write requests.csv (same columns as workload_driver.py output)
    csv_path = os.path.join(args.output_dir, "requests.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tenant_id", "request_idx", "ttft_ms", "tpot_ms",
            "mean_itl_ms", "e2e_ms", "num_input_tokens",
            "num_output_tokens", "start_time",
        ])
        for i, (req, m) in enumerate(zip(requests, all_metrics)):
            # tpot: e2e minus ttft, divided by output tokens
            n_out = m["num_output_tokens"]
            tpot = ((m["e2e_ms"] - m["ttft_ms"]) / n_out) if n_out > 0 else 0.0
            # start_time: send_time_us converted to seconds (Unix epoch)
            start_time = req["send_time_us"] / 1_000_000.0
            writer.writerow([
                req["tenant_id"],
                i,
                round(m["ttft_ms"], 2),
                round(tpot, 2),
                round(m["mean_itl_ms"], 2),
                round(m["e2e_ms"], 2),
                m["num_input_tokens"],
                m["num_output_tokens"],
                round(start_time, 3),
            ])
    print(f"  Wrote {csv_path} ({len(requests)} rows)")


if __name__ == "__main__":
    main()
