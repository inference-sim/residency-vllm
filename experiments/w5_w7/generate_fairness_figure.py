#!/usr/bin/env python3
"""Generate the fidelity-vs-load figure for the asymmetric fairness experiment.

Plots rho_mt (real and sim) vs aggregate rate for W5 and/or W7 sweeps.
Marks the saturation knee where mean E2E latency turns up sharply.

Usage:
    python3 generate_fairness_figure.py [--results-dir ../../results] [--workload w5|w7|both]

Output:
    results/w5_sweep/fig_rho_mt_vs_rate.pdf/.png
    results/w7_sweep/fig_rho_mt_vs_rate.pdf/.png  (if W7 data present)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Style (Wong colorblind-safe, matches paper campaigns) ---
mpl.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.10,
})

REAL_C = "#0072B2"   # blue
SIM_C = "#009E73"    # bluish green
KNEE_C = "#D55E00"   # vermillion
REF_C = "#333333"


def load_sweep_data(sweep_dir):
    """Load rho_mt and E2E latency data from all agg* cells in a sweep directory."""
    cells = []
    for cell_name in sorted(os.listdir(sweep_dir)):
        cell_dir = os.path.join(sweep_dir, cell_name)
        if not os.path.isdir(cell_dir) or not cell_name.startswith("agg"):
            continue

        rate = int(cell_name.replace("agg", ""))
        summary_path = os.path.join(cell_dir, "summary.json")
        sim_path = os.path.join(cell_dir, "sim.json")

        if not os.path.isfile(summary_path):
            continue

        real = json.load(open(summary_path))
        per_tenant = real.get("per_tenant", {})
        if not per_tenant:
            continue

        # Real rho_mt
        real_residencies = {t: v.get("residency_token_seconds", 0)
                           for t, v in per_tenant.items()}
        vals_r = [v for v in real_residencies.values() if v > 0]
        if len(vals_r) < 2:
            continue
        rho_real = max(vals_r) / min(vals_r)

        # Real mean E2E latency (for saturation knee detection)
        e2e_real = None
        overall = real.get("overall", {})
        if overall:
            lat = overall.get("latency", {})
            e2e_real = lat.get("e2e_ms", {}).get("median")
        if e2e_real is None:
            # Try computing from per_tenant
            e2es = []
            for v in per_tenant.values():
                e = v.get("e2e_ms", {}).get("median") or v.get("e2e_ms", {}).get("p50")
                if e is not None:
                    e2es.append(e)
            if e2es:
                e2e_real = sum(e2es) / len(e2es)

        # Sim rho_mt (if available)
        rho_sim = None
        if os.path.isfile(sim_path):
            sim = json.load(open(sim_path))
            tenants_sim = sim.get("tenants", sim.get("per_tenant", {}))
            sim_residencies = {}
            for t, v in tenants_sim.items():
                if isinstance(v, dict):
                    if "kv_time_token_us" in v:
                        sim_residencies[t] = v["kv_time_token_us"] / 1e6
                    elif "residency_token_seconds" in v:
                        sim_residencies[t] = v["residency_token_seconds"]
            vals_s = [v for v in sim_residencies.values() if v > 0]
            if len(vals_s) >= 2:
                rho_sim = max(vals_s) / min(vals_s)

        cells.append({
            "rate": rate,
            "rho_real": rho_real,
            "rho_sim": rho_sim,
            "e2e_real": e2e_real,
        })

    cells.sort(key=lambda c: c["rate"])
    return cells


def detect_saturation_knee(cells):
    """Detect the saturation knee as the rate where E2E latency gradient is steepest.

    Returns the rate at the knee, or None if insufficient data.
    """
    points = [(c["rate"], c["e2e_real"]) for c in cells if c["e2e_real"] is not None]
    if len(points) < 3:
        return None

    # Compute discrete second derivative (acceleration of latency)
    rates = [p[0] for p in points]
    latencies = [p[1] for p in points]

    max_accel = 0
    knee_rate = None
    for i in range(1, len(rates) - 1):
        dr1 = rates[i] - rates[i - 1]
        dr2 = rates[i + 1] - rates[i]
        dl1 = (latencies[i] - latencies[i - 1]) / dr1
        dl2 = (latencies[i + 1] - latencies[i]) / dr2
        accel = (dl2 - dl1) / ((dr1 + dr2) / 2)
        if accel > max_accel:
            max_accel = accel
            knee_rate = rates[i]

    return knee_rate


def plot_rho_mt_vs_rate(cells, sweep_dir, workload_label):
    """Generate the rho_mt vs aggregate rate figure."""
    rates = [c["rate"] for c in cells]
    rho_reals = [c["rho_real"] for c in cells]
    rho_sims = [c["rho_sim"] for c in cells]

    has_sim = any(r is not None for r in rho_sims)

    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)

    # Title and subtitle
    subtitle_map = {
        "W5": "5:1 arrival rate split, equal prompt lengths",
        "W7": "Equal arrival rates, 16:1 prompt length split (256 vs 4096 tokens)",
    }
    fig.suptitle(f"{workload_label}: Residency Disparity Fidelity vs Load",
                 fontsize=11, fontweight="bold")
    ax.set_title(subtitle_map.get(workload_label, ""), fontsize=8.5, color="#555555", pad=4)

    # Plot real rho_mt
    ax.plot(rates, rho_reals, "o-", color=REAL_C, markersize=7,
            linewidth=1.8, label=r"$\rho_{mt}$ real (vLLM)", zorder=3)

    # Plot sim rho_mt (if available)
    if has_sim:
        sim_rates = [c["rate"] for c in cells if c["rho_sim"] is not None]
        sim_vals = [c["rho_sim"] for c in cells if c["rho_sim"] is not None]
        ax.plot(sim_rates, sim_vals, "s--", color=SIM_C, markersize=6,
                linewidth=1.8, label=r"$\rho_{mt}$ sim (BLIS)", zorder=3)

    # Mark saturation knee
    knee = detect_saturation_knee(cells)
    if knee is not None:
        ax.axvline(knee, color=KNEE_C, ls=":", lw=1.2, zorder=2)
        ax.text(knee + 0.3, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 1 else max(rho_reals) * 0.95,
                f"saturation\nknee ≈ {knee}", fontsize=8, color=KNEE_C, va="top")

    # Reference line at theoretical ratio
    if workload_label == "W5":
        ax.axhline(5.0, color=REF_C, ls=(0, (4, 3)), lw=0.9, zorder=1, alpha=0.5)
        ax.text(rates[-1] + 0.3, 5.0, "5:1\ntheory", fontsize=7.5, color=REF_C,
                va="center", ha="left", alpha=0.7)

    ax.set_xlabel("aggregate rate (req/s)")
    ax.set_ylabel(r"disparity ratio $\rho_{mt}$")
    ax.set_xticks(rates)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # Set y-axis to show some context around the values
    ymin = min(rho_reals + [r for r in rho_sims if r]) * 0.9
    ymax = max(rho_reals + [r for r in rho_sims if r]) * 1.1
    ax.set_ylim(ymin, ymax)

    # Save
    out_dir = Path(sweep_dir)
    fig.savefig(out_dir / "fig_rho_mt_vs_rate.png")
    plt.close(fig)
    print(f"  Wrote: {out_dir}/fig_rho_mt_vs_rate.png")

    # Also generate the per-tenant absolute residency comparison (secondary figure)
    if has_sim:
        plot_absolute_comparison(cells, sweep_dir, workload_label)


def plot_absolute_comparison(cells, sweep_dir, workload_label):
    """Plot per-tenant absolute residency: real vs sim, grouped by rate."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)

    # Title and subtitle
    fig.suptitle(f"{workload_label}: Per-Tenant Absolute Residency",
                 fontsize=11, fontweight="bold")
    ax.set_title("Real vLLM (solid) vs simulated BLIS (dashed)",
                 fontsize=8.5, color="#555555", pad=4)

    rates = [c["rate"] for c in cells]

    # Load per-tenant data
    for cell in cells:
        cell_dir = os.path.join(sweep_dir, f"agg{cell['rate']}")
        summary = json.load(open(os.path.join(cell_dir, "summary.json")))
        sim_path = os.path.join(cell_dir, "sim.json")
        if not os.path.isfile(sim_path):
            continue
        sim = json.load(open(sim_path))

        per_tenant = summary.get("per_tenant", {})
        tenants_sim = sim.get("tenants", sim.get("per_tenant", {}))

        cell["real_A"] = per_tenant.get("tenantA", {}).get("residency_token_seconds", 0) / 1e6
        cell["real_B"] = per_tenant.get("tenantB", {}).get("residency_token_seconds", 0) / 1e6

        sim_A = tenants_sim.get("tenantA", {})
        sim_B = tenants_sim.get("tenantB", {})
        cell["sim_A"] = (sim_A.get("kv_time_token_us", 0) / 1e12) if "kv_time_token_us" in sim_A else sim_A.get("residency_token_seconds", 0) / 1e6
        cell["sim_B"] = (sim_B.get("kv_time_token_us", 0) / 1e12) if "kv_time_token_us" in sim_B else sim_B.get("residency_token_seconds", 0) / 1e6

    real_A = [c.get("real_A", 0) for c in cells]
    real_B = [c.get("real_B", 0) for c in cells]
    sim_A = [c.get("sim_A", 0) for c in cells]
    sim_B = [c.get("sim_B", 0) for c in cells]

    ax.plot(rates, real_B, "o-", color=REAL_C, markersize=6, lw=1.5, label="tenantB real")
    ax.plot(rates, sim_B, "s--", color=SIM_C, markersize=5, lw=1.5, label="tenantB sim")
    ax.plot(rates, real_A, "o-", color=REAL_C, markersize=6, lw=1.5, alpha=0.5, label="tenantA real")
    ax.plot(rates, sim_A, "s--", color=SIM_C, markersize=5, lw=1.5, alpha=0.5, label="tenantA sim")

    ax.set_xlabel("aggregate rate (req/s)")
    ax.set_ylabel("residency (M token-s)")
    ax.set_xticks(rates)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    out_dir = Path(sweep_dir)
    fig.savefig(out_dir / "fig_residency_absolute.png")
    plt.close(fig)
    print(f"  Wrote: {out_dir}/fig_residency_absolute.png")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "..", "..")
    default_results = os.path.join(repo_root, "results")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=default_results,
                        help="Root results directory")
    parser.add_argument("--workload", default="both", choices=["w5", "w7", "both"],
                        help="Which sweep to plot")
    args = parser.parse_args()

    for sweep, label in [("w5_sweep", "W5"), ("w7_sweep", "W7")]:
        if args.workload != "both" and sweep.split("_")[0] != args.workload:
            continue
        sweep_dir = os.path.join(args.results_dir, sweep)
        if not os.path.isdir(sweep_dir):
            print(f"  Skipping {label}: {sweep_dir} not found")
            continue

        print(f"=== {label} ===")
        cells = load_sweep_data(sweep_dir)
        if not cells:
            print(f"  No data found in {sweep_dir}")
            continue

        plot_rho_mt_vs_rate(cells, sweep_dir, label)
        print()


if __name__ == "__main__":
    main()
