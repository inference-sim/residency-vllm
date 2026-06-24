#!/usr/bin/env python3
"""Generate sweep figures from A/B experiment results.

Usage:
    python3 generate_figures.py [--results-dir ./results]

Reads summary.json files from:
    results/sweep_rate/agg_*/patched/summary.json
    results/sweep_rate/agg_*/vanilla/summary.json
    results/sweep_tenants/*T_agg_*/patched/summary.json
    results/sweep_tenants/*T_agg_*/vanilla/summary.json

Outputs:
    results/sweep_rate/figure_rate_sweep.png
    results/sweep_tenants/figure_tenant_sweep.png
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_sweep_data(sweep_dir, pattern, key_extractor):
    """Load patched/vanilla summary.json pairs from sweep subdirectories.

    Args:
        sweep_dir: Path to sweep_rate/ or sweep_tenants/
        pattern: Regex to match subdirectory names
        key_extractor: Function that extracts the sort key from a dirname

    Returns:
        List of (key, patched_summary, vanilla_summary) tuples, sorted by key.
    """
    data = []
    if not os.path.isdir(sweep_dir):
        return data

    for entry in os.listdir(sweep_dir):
        if not os.path.isdir(os.path.join(sweep_dir, entry)):
            continue
        match = re.match(pattern, entry)
        if not match:
            continue

        ppath = os.path.join(sweep_dir, entry, "patched", "summary.json")
        vpath = os.path.join(sweep_dir, entry, "vanilla", "summary.json")
        if not os.path.isfile(ppath) or not os.path.isfile(vpath):
            continue

        with open(ppath) as f:
            patched = json.load(f)
        with open(vpath) as f:
            vanilla = json.load(f)

        key = key_extractor(entry, match)
        data.append((key, patched, vanilla))

    data.sort(key=lambda x: x[0])
    return data


def avg_p50(summary, metric):
    """Average the p50 of a metric across all tenants.

    Supports both old format (p50 key) and new format (median key).
    """
    values = []
    for tenant_data in summary.get("per_tenant", {}).values():
        m = tenant_data.get(metric, {})
        val = m.get("p50", m.get("median"))
        if val is not None:
            values.append(val)
    return np.mean(values) if values else 0.0


def compute_overhead_pct(patched_summaries, vanilla_summaries):
    """Compute median E2E overhead % across all data points.

    Uses median to be robust against saturated-regime outliers where
    queueing amplifies small per-step differences.
    """
    overheads = []
    for p, v in zip(patched_summaries, vanilla_summaries):
        for tenant in p.get("per_tenant", {}):
            if tenant in v.get("per_tenant", {}):
                pe = p["per_tenant"][tenant]["e2e_ms"]
                ve = v["per_tenant"][tenant]["e2e_ms"]
                p_val = pe.get("p50", pe.get("median", 0))
                v_val = ve.get("p50", ve.get("median", 0))
                if v_val > 0:
                    overheads.append((p_val - v_val) / v_val * 100)
    if overheads:
        return np.median(overheads)
    return 0.0


def generate_rate_sweep_figure(results_dir):
    """Generate Figure 1: Latency vs Per-Tenant Request Rate."""
    sweep_dir = os.path.join(results_dir, "sweep_rate")

    def key_extractor(dirname, match):
        # Extract aggregate rps from "agg_Xrps" and compute per-tenant rate
        agg = int(re.search(r"agg_(\d+)rps", dirname).group(1))
        return agg

    data = load_sweep_data(sweep_dir, r"agg_\d+rps", key_extractor)
    if not data:
        print("No rate sweep data found, skipping figure.", file=sys.stderr)
        return None

    # Determine number of tenants from first data point
    first_patched = data[0][1]
    n_tenants = len(first_patched.get("per_tenant", {}))

    # Extract metrics
    rates = []
    patched_ttft, patched_itl, patched_e2e = [], [], []
    vanilla_ttft, vanilla_itl, vanilla_e2e = [], [], []
    patched_summaries, vanilla_summaries = [], []

    for agg_rps, patched, vanilla in data:
        per_tenant_rate = agg_rps / n_tenants
        rates.append(per_tenant_rate)
        patched_ttft.append(avg_p50(patched, "ttft_ms"))
        vanilla_ttft.append(avg_p50(vanilla, "ttft_ms"))
        patched_itl.append(avg_p50(patched, "itl_ms"))
        vanilla_itl.append(avg_p50(vanilla, "itl_ms"))
        patched_e2e.append(avg_p50(patched, "e2e_ms"))
        vanilla_e2e.append(avg_p50(vanilla, "e2e_ms"))
        patched_summaries.append(patched)
        vanilla_summaries.append(vanilla)

    overhead = compute_overhead_pct(patched_summaries, vanilla_summaries)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    panels = [
        (patched_ttft, vanilla_ttft, "Time to First Token", "TTFT p50 (ms)"),
        (patched_itl, vanilla_itl, "Inter-Token Latency", "ITL p50 (ms)"),
        (patched_e2e, vanilla_e2e, "End-to-End Latency", "E2E p50 (ms)"),
    ]

    for ax, (p_data, v_data, title, ylabel) in zip(axes, panels):
        ax.plot(rates, p_data, "o-", color="#d62728", linewidth=2,
                markersize=8, label="vLLM + Residency")
        ax.plot(rates, v_data, "s--", color="#1f77b4", linewidth=2,
                markersize=8, label="Stock vLLM")
        ax.set_xlabel("Per-Tenant Request Rate (req/s)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rates)

    duration = first_patched.get("config", {}).get("duration", "?")
    model = first_patched.get("config", {}).get("model", "?")

    fig.suptitle(
        f"Figure 1: Latency vs Per-Tenant Request Rate\n"
        f"({n_tenants} tenants, {model}, {duration // 60}-min Poisson workload)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, -0.02,
        f"Both variants scale identically across all load levels. "
        f"The residency instrumentation adds ~{abs(overhead):.0f}% E2E overhead.",
        ha="center", fontsize=11, style="italic", color="#444444",
    )

    plt.tight_layout()
    out_path = os.path.join(sweep_dir, "figure_rate_sweep.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def generate_tenant_sweep_figure(results_dir):
    """Generate Figure 2: Latency vs Number of Tenants."""
    sweep_dir = os.path.join(results_dir, "sweep_tenants")

    def key_extractor(dirname, match):
        # Extract tenant count from "NT_agg_Yrps"
        return int(re.search(r"(\d+)T_", dirname).group(1))

    data = load_sweep_data(sweep_dir, r"\d+T_agg_\d+rps", key_extractor)
    if not data:
        print("No tenant sweep data found, skipping figure.", file=sys.stderr)
        return None

    # Extract per-tenant rate from first data point
    first_patched = data[0][1]
    per_tenant_rate = first_patched.get("config", {}).get("rate",
                      first_patched.get("config", {}).get("rate_per_tenant", "?"))

    # Extract metrics
    tenants = []
    patched_ttft, patched_itl, patched_e2e = [], [], []
    vanilla_ttft, vanilla_itl, vanilla_e2e = [], [], []
    patched_summaries, vanilla_summaries = [], []

    for n_tenants, patched, vanilla in data:
        tenants.append(n_tenants)
        patched_ttft.append(avg_p50(patched, "ttft_ms"))
        vanilla_ttft.append(avg_p50(vanilla, "ttft_ms"))
        patched_itl.append(avg_p50(patched, "itl_ms"))
        vanilla_itl.append(avg_p50(vanilla, "itl_ms"))
        patched_e2e.append(avg_p50(patched, "e2e_ms"))
        vanilla_e2e.append(avg_p50(vanilla, "e2e_ms"))
        patched_summaries.append(patched)
        vanilla_summaries.append(vanilla)

    overhead = compute_overhead_pct(patched_summaries, vanilla_summaries)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    panels = [
        (patched_ttft, vanilla_ttft, "Time to First Token", "TTFT p50 (ms)"),
        (patched_itl, vanilla_itl, "Inter-Token Latency", "ITL p50 (ms)"),
        (patched_e2e, vanilla_e2e, "End-to-End Latency", "E2E p50 (ms)"),
    ]

    for ax, (p_data, v_data, title, ylabel) in zip(axes, panels):
        ax.plot(tenants, p_data, "o-", color="#d62728", linewidth=2,
                markersize=8, label="vLLM + Residency")
        ax.plot(tenants, v_data, "s--", color="#1f77b4", linewidth=2,
                markersize=8, label="Stock vLLM")
        ax.set_xlabel("Number of Tenants", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(tenants)

    duration = first_patched.get("config", {}).get("duration", "?")
    model = first_patched.get("config", {}).get("model", "?")

    fig.suptitle(
        f"Figure 2: Latency vs Number of Tenants\n"
        f"({per_tenant_rate} req/s per tenant, {model}, {duration // 60}-min Poisson workload)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, -0.02,
        f"Latency grows with tenant count (more concurrent decodes competing "
        f"for GPU), but both variants track identically — the residency counter "
        f"adds ~{abs(overhead):.0f}% E2E overhead.",
        ha="center", fontsize=11, style="italic", color="#444444",
    )

    plt.tight_layout()
    out_path = os.path.join(sweep_dir, "figure_tenant_sweep.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "..", "..")
    default_results = os.path.join(repo_root, "results")
    parser.add_argument("--results-dir", default=default_results,
                        help="Root results directory")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print("Generating sweep figures...")

    path1 = generate_rate_sweep_figure(results_dir)
    if path1:
        print(f"  Saved: {path1}")

    path2 = generate_tenant_sweep_figure(results_dir)
    if path2:
        print(f"  Saved: {path2}")

    if not path1 and not path2:
        print("No sweep data found. Run experiments first.", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
