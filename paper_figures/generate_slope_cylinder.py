#!/usr/bin/env python3
"""Focused slope chart: Cylinder Flow only, 4 architectures (no DPOT).
Pretrained → DFT → PhysGuard trajectory for low-frequency error."""

import os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========================== CONFIG ==========================
RESULTS_DIR = "./results"
OUTPUT_DIR  = "./figures"

DATASET = "001-cylinder"

ARCHS      = ["cno", "deeponet", "fno", "transolver"]  # No DPOT
ARCH_NAMES = {
    "cno": "CNO", "deeponet": "DeepONet",
    "fno": "FNO", "transolver": "Transolver",
}

METHOD_ORDER = ["pretrained", "dft", "l2ft", "ewcft", "nsft"]
METHOD_NAMES = {
    "pretrained": "Pretrained", "dft": "DFT", "l2ft": "L2-SP",
    "ewcft": "EWC", "nsft": "PhysGuard",
}

# ========================== LOG PARSING ==========================

def parse_log_path(log_path):
    parts = log_path.replace("\\", "/").split("/")
    try:
        idx = parts.index("results")
    except ValueError:
        return None, None, None, None
    dataset = parts[idx + 1]
    arch    = parts[idx + 2]
    method_dir = parts[idx + 3]
    method = None
    for suffix in ["pretrained", "nsft", "ewcft", "l2ft", "dft"]:
        if f"_{suffix}" in method_dir:
            method = suffix
            break
    date_str = None
    for p in parts[idx + 3:]:
        if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", p):
            date_str = p
            break
    return dataset, arch, method, date_str


def parse_nsft_line(line):
    metrics = {}
    chunks = line.split(", ")
    mapping = [
        ("normalized mse loss", "nmse"),
        ("rel low f error",     "rel_low_f"),
        ("rel mid f error",     "rel_mid_f"),
        ("rel high f error",    "rel_high_f"),
        ("rel l2 error",        "rel_l2"),
        ("low f error",         "low_f"),
        ("mid f error",         "mid_f"),
        ("high f error",        "high_f"),
        ("freq error",          "freq_err"),
        ("ke error",            "ke_err"),
        ("f error",             "f_err"),
        ("rmse",                "rmse"),
        ("mae",                 "mae"),
        ("r2",                  "r2"),
    ]
    for chunk in chunks:
        chunk = chunk.strip()
        for text_key, metric_key in mapping:
            if chunk.startswith(text_key):
                m = re.search(r":\s*([\d.eE+-]+)", chunk)
                if m:
                    try:
                        metrics[metric_key] = float(m.group(1))
                    except ValueError:
                        pass
                break
    return metrics if "rel_low_f" in metrics else None


def find_best_iteration(content):
    m = re.search(r"best iteration[=:]\s*(\d+)", content, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"Best iteration:\s*(\d+)", content)
    if m:
        return int(m.group(1))
    return None


def parse_gpus_block(lines, target_iter):
    for i, line in enumerate(lines):
        if f"Iteration {target_iter}," in line:
            block = "\n".join(lines[i : min(i + 5, len(lines))])
            if "rel_low_f=" in block:
                metrics = {}
                for key in ["low_f", "mid_f", "high_f",
                             "rel_low_f", "rel_mid_f", "rel_high_f",
                             "rmse", "rel_l2", "r2"]:
                    m = re.search(rf"(?<![a-z_]){key}=([\d.eE+-]+)", block)
                    if m:
                        metrics[key] = float(m.group(1))
                if "rel_low_f" in metrics:
                    return metrics
    return None


def parse_nsft_block(lines, target_iter):
    for i, line in enumerate(lines):
        if f"Iteration {target_iter}," in line:
            for j in range(i, min(i + 5, len(lines))):
                if "rel low f error" in lines[j]:
                    return parse_nsft_line(lines[j])
    return None


def parse_metrics_from_log(log_path):
    with open(log_path, "r") as f:
        content = f.read()
    lines = content.split("\n")
    best_iter = find_best_iteration(content)

    if "rel_low_f=" in content:
        if best_iter is not None:
            metrics = parse_gpus_block(lines, best_iter)
            if metrics:
                return metrics
        last_idx = None
        for i, line in enumerate(lines):
            if "rel_low_f=" in line:
                last_idx = i
        if last_idx is None:
            return None
        block = "\n".join(lines[max(0, last_idx - 3) : last_idx + 1])
        metrics = {}
        for key in ["low_f", "mid_f", "high_f",
                     "rel_low_f", "rel_mid_f", "rel_high_f",
                     "rmse", "rel_l2", "r2"]:
            m = re.search(rf"(?<![a-z_]){key}=([\d.eE+-]+)", block)
            if m:
                metrics[key] = float(m.group(1))
        return metrics if "rel_low_f" in metrics else None

    if "rel low f error" in content:
        if best_iter is not None:
            metrics = parse_nsft_block(lines, best_iter)
            if metrics:
                return metrics
        last_val_line = None
        for line in lines:
            if "rel low f error" in line:
                last_val_line = line
        if last_val_line is None:
            return None
        return parse_nsft_line(last_val_line)

    return None


def collect_all_data():
    logs = []
    for root, dirs, files in os.walk(RESULTS_DIR):
        if "training.log" in files:
            logs.append(os.path.join(root, "training.log"))
    groups = defaultdict(list)
    for log_path in logs:
        dataset, arch, method, date_str = parse_log_path(log_path)
        if dataset == DATASET and arch in ARCHS and method in METHOD_ORDER:
            groups[(dataset, arch, method)].append((date_str or "", log_path))
    data = {}
    for key, runs in groups.items():
        runs.sort(key=lambda x: x[0], reverse=True)
        metrics = parse_metrics_from_log(runs[0][1])
        if metrics:
            data[key] = metrics
    return data


# ========================== SLOPE CHART ==========================

def repel_labels(positions, min_gap=0.03):
    """Adjust y positions to avoid overlap. positions = list of (y, index).
    Returns adjusted y values keyed by original index."""
    sorted_pos = sorted(positions, key=lambda x: x[0])
    adjusted = []
    for y, idx in sorted_pos:
        if adjusted and y - adjusted[-1][0] < min_gap:
            y = adjusted[-1][0] + min_gap
        adjusted.append((y, idx))
    return {idx: y for y, idx in adjusted}


def make_slope_chart(data):
    """Single-panel slope chart for Cylinder Flow.
    Each line = one architecture, X = Pretrained → DFT → PhysGuard."""

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 12,
        'axes.titlesize': 15,
        'axes.labelsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    arch_colors = {
        "CNO":        "#e74c3c",
        "DeepONet":   "#9b59b6",
        "FNO":        "#3498db",
        "Transolver": "#f39c12",
    }

    stages = ["pretrained", "dft", "nsft"]
    stage_labels = ["Pretrained", "DFT", "PhysGuard"]
    x_positions = [0, 1, 2]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))

    # Collect all endpoint values for label repulsion
    left_labels = []   # (y_val, arch_index)
    right_labels = []  # (y_val, arch_index)
    all_vals = {}      # arch -> [pre, dft, nsft]

    for ai, ar in enumerate(ARCHS):
        ar_name = ARCH_NAMES[ar]
        vals = []
        for st in stages:
            rec = data.get((DATASET, ar, st))
            vals.append(rec["rel_low_f"] if rec and "rel_low_f" in rec else np.nan)
        all_vals[ar] = vals
        if not np.isnan(vals[0]):
            left_labels.append((vals[0], ai))
        if not np.isnan(vals[2]):
            right_labels.append((vals[2], ai))

    # Compute repelled positions (min gap ~0.035 in data coords)
    left_adj = repel_labels(left_labels, min_gap=0.038)
    right_adj = repel_labels(right_labels, min_gap=0.045)

    for ai, ar in enumerate(ARCHS):
        ar_name = ARCH_NAMES[ar]
        vals = all_vals[ar]
        color = arch_colors[ar_name]
        marker = {"CNO": "o", "DeepONet": "s", "FNO": "^", "Transolver": "v"}[ar_name]

        valid_x = [x for x, v in zip(x_positions, vals) if not np.isnan(v)]
        valid_v = [v for v in vals if not np.isnan(v)]

        # Main line
        ax.plot(valid_x, valid_v, '-', color=color, linewidth=3,
                alpha=0.85, zorder=4)
        ax.scatter(valid_x, valid_v, s=140, marker=marker, color=color,
                   edgecolors="white", linewidths=2, zorder=5)

        # Left labels (Pretrained value) — repelled
        if not np.isnan(vals[0]):
            label_y = left_adj[ai]
            ax.annotate(f"{vals[0]:.3f}", xy=(0, vals[0]),
                        xytext=(-0.12, label_y),
                        ha="right", va="center", fontsize=10.5,
                        color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=color,
                                        alpha=0.3, lw=0.8)
                        if abs(label_y - vals[0]) > 0.01 else None)

        # Right labels (PhysGuard value + arch name) — repelled
        if not np.isnan(vals[2]):
            label_y = right_adj[ai]
            ax.annotate(f"{vals[2]:.3f}  {ar_name}", xy=(2, vals[2]),
                        xytext=(2.12, label_y),
                        ha="left", va="center", fontsize=11,
                        color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=color,
                                        alpha=0.3, lw=0.8)
                        if abs(label_y - vals[2]) > 0.01 else None)

        # Δ% badge — place at the midpoint of the Pre→PG line, offset left
        if len(valid_v) == 3:
            pre_val = valid_v[0]
            pg_val = valid_v[2]
            delta_pct = (pg_val - pre_val) / pre_val * 100
            mid_y = (pre_val + pg_val) / 2
            # Stagger horizontally by arch index to avoid overlap
            badge_x = 0.35 + ai * 0.15
            ax.text(badge_x, mid_y, f"{delta_pct:+.1f}%",
                    fontsize=9.5, color=color, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=color, alpha=0.9, linewidth=1.2),
                    zorder=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(stage_labels, fontsize=14, fontweight="bold")
    ax.set_ylabel("Relative Low-Frequency Error ↓", fontsize=13)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_xlim(-0.5, 3.5)

    # Pad y-axis
    all_y = [v for ar in ARCHS for v in all_vals[ar] if not np.isnan(v)]
    y_margin = (max(all_y) - min(all_y)) * 0.1
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    ax.set_title("Cylinder Flow: Low-Frequency Error Trajectory",
                 fontsize=15, fontweight="bold", pad=12)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "006-slope-cylinder.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    print("Collecting Cylinder Flow data...")
    data = collect_all_data()
    print(f"  Found {len(data)} entries")

    # Print summary
    print("\n  Cylinder Flow (rel_low_f at best iteration):")
    for ar in ARCHS:
        pre = data.get((DATASET, ar, "pretrained"), {}).get("rel_low_f", float("nan"))
        dft = data.get((DATASET, ar, "dft"), {}).get("rel_low_f", float("nan"))
        nsft = data.get((DATASET, ar, "nsft"), {}).get("rel_low_f", float("nan"))
        delta = (nsft - pre) / pre * 100 if pre > 0 else float("nan")
        print(f"    {ARCH_NAMES[ar]:12s}  Pre={pre:.3f}  DFT={dft:.3f}  PhysGuard={nsft:.3f}  (Δ={delta:+.1f}%)")

    print("\nGenerating slope chart...")
    make_slope_chart(data)
    print("Done!")


if __name__ == "__main__":
    main()
