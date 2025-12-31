#!/usr/bin/env python3
"""Generate 5 alternative frequency-domain visualizations for PhysGuard paper.

Scheme A: Δ% relative to Pretrained (forgetting vs. improvement)
Scheme B: Frequency band error decomposition (stacked bar)
Scheme C: Paired slope chart (Pretrained → DFT → PhysGuard)
Scheme D: Multi-architecture frequency profile (3-band comparison)
Scheme E: Win/Lose heatmap (PhysGuard vs DFT improvement %)
"""

import os, re, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========================== CONFIG ==========================
RESULTS_DIR = "./results"
OUTPUT_DIR  = "./figures"

DATASETS     = ["001-cylinder", "002-control_cylinder", "003-combustion"]
DATASET_NAMES = {
    "001-cylinder":          "Cylinder Flow",
    "002-control_cylinder":  "Controlled Cylinder",
    "003-combustion":        "Combustion",
}

ARCHS      = ["cno", "deeponet", "dpot", "fno", "transolver"]
ARCH_NAMES = {
    "cno": "CNO", "deeponet": "DeepONet", "dpot": "DPOT",
    "fno": "FNO", "transolver": "Transolver",
}

METHOD_ORDER = ["pretrained", "dft", "l2ft", "ewcft", "nsft"]
METHOD_NAMES = {
    "pretrained": "Pretrained", "dft": "DFT", "l2ft": "L2-SP",
    "ewcft": "EWC", "nsft": "PhysGuard",
}
# Methods to compare (excluding pretrained baseline)
FT_METHODS = ["dft", "l2ft", "ewcft", "nsft"]

COLORS = {
    "Pretrained": "#7f8c8d",
    "DFT":        "#27ae60",
    "L2-SP":      "#e74c3c",
    "EWC":        "#3498db",
    "PhysGuard":  "#f39c12",
}

MARKERS_ARCH = {
    "CNO": "o", "DeepONet": "s", "DPOT": "D", "FNO": "^", "Transolver": "v",
}

FREQ_BANDS = ["Low", "Mid", "High"]
REL_KEYS   = ["rel_low_f", "rel_mid_f", "rel_high_f"]

# ========================== LOG PARSING (from v2) ==========================

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
                             "rmse", "rel_l2", "r2", "f_err", "freq_err", "ke_err"]:
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
                     "rmse", "rel_l2", "r2", "f_err", "freq_err", "ke_err"]:
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
        if dataset in DATASETS and arch in ARCHS and method in METHOD_ORDER:
            groups[(dataset, arch, method)].append((date_str or "", log_path))
    data = {}
    for key, runs in groups.items():
        runs.sort(key=lambda x: x[0], reverse=True)
        metrics = parse_metrics_from_log(runs[0][1])
        if metrics:
            data[key] = metrics
    return data


# ========================== STYLE HELPERS ==========================

def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ========================== SCHEME A: Δ% relative to Pretrained ==========================

def scheme_A_delta_pct(data):
    """Bar chart showing Δ% change in rel_low_f compared to Pretrained.
    Positive = forgetting (worse), Negative = improvement (better).
    Highlights catastrophic forgetting cases and PhysGuard protection."""

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    bar_width = 0.18
    x_base = np.arange(len(ARCHS))

    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        ax.set_title(DATASET_NAMES[ds], fontweight="bold", fontsize=14)
        ax.axhline(0, color="black", linewidth=1.2, zorder=2)

        for mi, mt in enumerate(FT_METHODS):
            deltas = []
            for ar in ARCHS:
                pre = data.get((ds, ar, "pretrained"))
                cur = data.get((ds, ar, mt))
                if pre and cur and "rel_low_f" in pre and "rel_low_f" in cur:
                    pct = (cur["rel_low_f"] - pre["rel_low_f"]) / pre["rel_low_f"] * 100
                    deltas.append(pct)
                else:
                    deltas.append(0)

            offset = (mi - len(FT_METHODS) / 2 + 0.5) * bar_width
            color = COLORS[METHOD_NAMES[mt]]

            # Color bars: red shade if positive (forgetting), green shade if negative (improvement)
            bar_colors = []
            for d in deltas:
                if d > 0:
                    bar_colors.append('#e74c3c')  # Forgetting = red
                else:
                    bar_colors.append(color)

            bars = ax.bar(x_base + offset, deltas, bar_width * 0.88,
                          color=color, edgecolor="white", linewidth=0.8,
                          label=METHOD_NAMES[mt] if di == 0 else None, zorder=3)

            # Value labels
            for bar, d in zip(bars, deltas):
                if abs(d) > 1:
                    va = "bottom" if d > 0 else "top"
                    y_offset = 1.5 if d > 0 else -1.5
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + y_offset,
                            f"{d:+.1f}%", ha="center", va=va,
                            fontsize=7.5, fontweight="bold" if abs(d) > 10 else "normal",
                            color="#333")

        ax.set_xticks(x_base)
        ax.set_xticklabels([ARCH_NAMES[a] for a in ARCHS], fontsize=11)
        ax.set_ylabel("Δ Low-Freq Error vs. Pretrained (%)" if di == 0 else "", fontsize=11)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        # Add shaded regions
        ylim = ax.get_ylim()
        ax.fill_between([-0.6, len(ARCHS) - 0.3], 0, max(ylim[1], 5),
                         color='#e74c3c', alpha=0.04, zorder=0)
        ax.fill_between([-0.6, len(ARCHS) - 0.3], min(ylim[0], -5), 0,
                         color='#27ae60', alpha=0.04, zorder=0)
        # Labels for regions
        ax.text(len(ARCHS) - 0.5, max(ylim[1] * 0.85, 3), "← Forgetting",
                ha="right", va="top", fontsize=9, color="#e74c3c", fontstyle="italic")
        ax.text(len(ARCHS) - 0.5, min(ylim[0] * 0.85, -3), "← Improved",
                ha="right", va="bottom", fontsize=9, color="#27ae60", fontstyle="italic")

        ax.set_xlim(-0.6, len(ARCHS) - 0.3)

    fig.legend(loc="lower center", ncol=4, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, -0.06),
               edgecolor="#cccccc")
    fig.suptitle("Change in Low-Frequency Error Relative to Pretrained Model\n"
                 "↑ Positive = Forgetting Physics  |  ↓ Negative = Better Preservation",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "006-scheme-A-delta-pct.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== SCHEME B: Frequency Band Decomposition ==========================

def scheme_B_stacked_bar(data):
    """Stacked bar showing Low/Mid/High band contributions.
    Each bar's total height = sum of 3 bands (≈ total rel freq error).
    Shows that PhysGuard selectively reduces low-freq while leaving mid/high similar."""

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    bar_width = 0.15
    x_base = np.arange(len(ARCHS))
    band_colors = {"Low": "#e74c3c", "Mid": "#f39c12", "High": "#3498db"}

    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        ax.set_title(DATASET_NAMES[ds], fontweight="bold", fontsize=14)

        for mi, mt in enumerate(METHOD_ORDER):
            lows, mids, highs = [], [], []
            for ar in ARCHS:
                rec = data.get((ds, ar, mt))
                lows.append(rec.get("rel_low_f", 0) if rec else 0)
                mids.append(rec.get("rel_mid_f", 0) if rec else 0)
                highs.append(rec.get("rel_high_f", 0) if rec else 0)

            offset = (mi - len(METHOD_ORDER) / 2 + 0.5) * bar_width
            x_pos = x_base + offset

            # Stacked bars: Low on bottom, Mid in middle, High on top
            alpha = 1.0 if mt == "nsft" else 0.65
            edge = "black" if mt == "nsft" else "white"
            lw = 1.2 if mt == "nsft" else 0.5

            b1 = ax.bar(x_pos, lows, bar_width * 0.88, color=band_colors["Low"],
                         alpha=alpha, edgecolor=edge, linewidth=lw, zorder=3)
            b2 = ax.bar(x_pos, mids, bar_width * 0.88, bottom=lows,
                         color=band_colors["Mid"], alpha=alpha, edgecolor=edge,
                         linewidth=lw, zorder=3)
            bottoms2 = [l + m for l, m in zip(lows, mids)]
            b3 = ax.bar(x_pos, highs, bar_width * 0.88, bottom=bottoms2,
                         color=band_colors["High"], alpha=alpha, edgecolor=edge,
                         linewidth=lw, zorder=3)

            # Method label on top of the bar group (only for first arch)
            if True:
                total = [l + m + h for l, m, h in zip(lows, mids, highs)]
                for xi, t in enumerate(total):
                    if t > 0 and xi == 0:  # label above first arch's bar
                        ax.text(x_pos[xi], t + 0.02, METHOD_NAMES[mt],
                                ha="center", va="bottom", fontsize=6.5,
                                rotation=60, color="#555")

        ax.set_xticks(x_base)
        ax.set_xticklabels([ARCH_NAMES[a] for a in ARCHS], fontsize=11)
        ax.set_ylabel("Relative Frequency Error" if di == 0 else "", fontsize=11)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    # Legend for frequency bands + method highlight
    band_handles = [mpatches.Patch(color=band_colors[b], label=f"{b}-Freq") for b in FREQ_BANDS]
    highlight_handle = mpatches.Patch(facecolor="white", edgecolor="black",
                                       linewidth=1.5, label="PhysGuard (black border)")
    fig.legend(handles=band_handles + [highlight_handle],
               loc="lower center", ncol=4, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.06), edgecolor="#cccccc")
    fig.suptitle("Frequency Error Decomposition: Low / Mid / High Bands\n"
                 "PhysGuard (black border) selectively reduces low-frequency error",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "006-scheme-B-stacked.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== SCHEME C: Paired Slope Chart ==========================

def scheme_C_slope(data):
    """Slope chart: each line = one architecture.
    X-axis: Pretrained → DFT → PhysGuard (3 stages)
    Y-axis: rel_low_f
    Clearly shows direction of change for each architecture."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    arch_colors = {
        "CNO": "#e74c3c", "DeepONet": "#9b59b6", "DPOT": "#2ecc71",
        "FNO": "#3498db", "Transolver": "#f39c12",
    }

    stages = ["pretrained", "dft", "nsft"]
    stage_labels = ["Pretrained", "DFT", "PhysGuard"]
    x_positions = [0, 1, 2]

    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        ax.set_title(DATASET_NAMES[ds], fontweight="bold", fontsize=14)

        for ar in ARCHS:
            ar_name = ARCH_NAMES[ar]
            vals = []
            for st in stages:
                rec = data.get((ds, ar, st))
                vals.append(rec["rel_low_f"] if rec and "rel_low_f" in rec else np.nan)

            color = arch_colors[ar_name]
            marker = MARKERS_ARCH[ar_name]

            # Draw connected lines
            valid_x = [x for x, v in zip(x_positions, vals) if not np.isnan(v)]
            valid_v = [v for v in vals if not np.isnan(v)]

            ax.plot(valid_x, valid_v, '-', color=color, linewidth=2.5,
                    alpha=0.8, zorder=4)
            ax.scatter(valid_x, valid_v, s=100, marker=marker, color=color,
                       edgecolors="white", linewidths=1.5, zorder=5)

            # Annotate values at endpoints
            for xi, vi in zip(valid_x, valid_v):
                if xi == 0:  # Pretrained: label on left
                    ax.text(xi - 0.08, vi, f"{vi:.3f}", ha="right", va="center",
                            fontsize=8, color=color, fontweight="bold")
                elif xi == 2:  # PhysGuard: label on right
                    ax.text(xi + 0.08, vi, f"{vi:.3f}", ha="left", va="center",
                            fontsize=8, color=color, fontweight="bold")

            # Label architecture name near the rightmost point
            if valid_v:
                last_v = valid_v[-1]
                last_x = valid_x[-1]
                ax.text(last_x + 0.15, last_v, ar_name, ha="left", va="center",
                        fontsize=9.5, color=color, fontweight="bold")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(stage_labels, fontsize=12, fontweight="bold")
        ax.set_ylabel("Relative Low-Freq Error ↓" if di == 0 else "", fontsize=11)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.set_xlim(-0.4, 2.7)

        # Shade the improvement zone
        ylim = ax.get_ylim()
        ax.annotate("", xy=(2, ylim[0] + (ylim[1] - ylim[0]) * 0.05),
                     xytext=(0, ylim[0] + (ylim[1] - ylim[0]) * 0.05),
                     arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2, alpha=0.4))

    fig.suptitle("Low-Frequency Error Trajectory: Pretrained → DFT → PhysGuard\n"
                 "Downward slope = better low-frequency preservation",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "006-scheme-C-slope.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== SCHEME D: Multi-Architecture Frequency Profile ==========================

def scheme_D_freq_profile(data):
    """For each architecture, plot 3-band (Low/Mid/High) error profile.
    Compares Pretrained vs DFT vs PhysGuard across frequency bands.
    Shows that PhysGuard selectively protects low-freq while keeping mid/high similar to DFT."""

    fig, axes = plt.subplots(len(ARCHS), 3, figsize=(18, 4 * len(ARCHS)),
                              sharex=True)

    band_x = np.arange(3)  # Low, Mid, High
    methods_to_show = ["pretrained", "dft", "nsft"]
    method_labels = ["Pretrained", "DFT", "PhysGuard"]
    line_styles = {"pretrained": "--", "dft": "-", "nsft": "-"}
    line_widths = {"pretrained": 2, "dft": 2, "nsft": 3}

    for ai, ar in enumerate(ARCHS):
        ar_name = ARCH_NAMES[ar]

        for di, ds in enumerate(DATASETS):
            ax = axes[ai, di] if len(ARCHS) > 1 else axes[di]

            for mt in methods_to_show:
                rec = data.get((ds, ar, mt))
                if rec:
                    vals = [rec.get(k, np.nan) for k in REL_KEYS]
                    color = COLORS[METHOD_NAMES[mt]]
                    ax.plot(band_x, vals, line_styles[mt],
                            color=color, linewidth=line_widths[mt],
                            marker="o", markersize=8, markeredgecolor="white",
                            markeredgewidth=1.5,
                            label=METHOD_NAMES[mt] if ai == 0 and di == 0 else None,
                            zorder=5 if mt == "nsft" else 3)

                    # Value annotation
                    for xi, vi in zip(band_x, vals):
                        if not np.isnan(vi):
                            ax.text(xi, vi + 0.02, f"{vi:.3f}", ha="center",
                                    va="bottom", fontsize=7.5, color=color)

            # Highlight the low-freq improvement
            pre = data.get((ds, ar, "pretrained"))
            nsft = data.get((ds, ar, "nsft"))
            if pre and nsft:
                pre_low = pre.get("rel_low_f", 0)
                nsft_low = nsft.get("rel_low_f", 0)
                if pre_low > nsft_low:
                    ax.annotate("", xy=(0, nsft_low), xytext=(0, pre_low),
                                arrowprops=dict(arrowstyle="->", color="#27ae60",
                                                lw=2, alpha=0.6))

            ax.set_xticks(band_x)
            ax.set_xticklabels(FREQ_BANDS if ai == len(ARCHS) - 1 else [], fontsize=10)
            if di == 0:
                ax.set_ylabel(f"{ar_name}", fontsize=12, fontweight="bold")
            if ai == 0:
                ax.set_title(DATASET_NAMES[ds], fontweight="bold", fontsize=13)

            ax.grid(axis="y", alpha=0.2, linestyle="--")
            ax.set_xlim(-0.3, 2.3)

    fig.legend(loc="lower center", ncol=3, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, -0.03), edgecolor="#cccccc")
    fig.suptitle("Frequency Error Profile across Bands\n"
                 "PhysGuard specifically reduces low-frequency error while preserving mid/high-frequency behavior",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "006-scheme-D-freq-profile.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== SCHEME E: Win/Lose Heatmap ==========================

def scheme_E_winlose(data):
    """Heatmap showing PhysGuard improvement % over DFT in rel_low_f.
    Green = PhysGuard better, Red = DFT better.
    Clean single-figure summary of where PhysGuard wins/loses."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Rows: datasets × archs, cols: low/mid/high bands
    row_labels = []
    mat = np.full((len(DATASETS) * len(ARCHS), 3), np.nan)

    for di, ds in enumerate(DATASETS):
        for ai, ar in enumerate(ARCHS):
            ri = di * len(ARCHS) + ai
            row_labels.append(f"{DATASET_NAMES[ds]}\n{ARCH_NAMES[ar]}")

            dft_rec = data.get((ds, ar, "dft"))
            nsft_rec = data.get((ds, ar, "nsft"))

            if dft_rec and nsft_rec:
                for bi, key in enumerate(REL_KEYS):
                    dft_val = dft_rec.get(key, np.nan)
                    nsft_val = nsft_rec.get(key, np.nan)
                    if not np.isnan(dft_val) and not np.isnan(nsft_val) and dft_val > 0:
                        mat[ri, bi] = (nsft_val - dft_val) / dft_val * 100

    # Diverging colormap: red (PhysGuard worse) → white (equal) → green (PhysGuard better)
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    vmax = min(vmax, 50)  # cap at ±50%

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Custom diverging: green (negative=better) to red (positive=worse)
    # Note: negative means PhysGuard is better (lower error)
    cmap = plt.cm.RdYlGn_r  # Red=positive(worse), Green=negative(better)
    # Actually we want: negative = PhysGuard better = green, positive = PhysGuard worse = red
    # RdYlGn: Red at low, Green at high → we need to reverse the sign
    # Let's use RdYlGn with the sign convention: plot -mat so green = better
    cmap = plt.cm.RdYlGn  # Green at high (positive = better for -mat)

    im = ax.imshow(-mat, cmap=cmap, aspect="auto", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))

    # Annotate each cell
    for ri in range(mat.shape[0]):
        for bi in range(3):
            val = mat[ri, bi]
            if not np.isnan(val):
                # Format with sign
                text = f"{val:+.1f}%"
                text_color = "white" if abs(val) > vmax * 0.65 else "black"
                weight = "bold" if abs(val) > 10 else "normal"
                ax.text(bi, ri, text, ha="center", va="center",
                        fontsize=10, fontweight=weight, color=text_color)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Low-Freq", "Mid-Freq", "High-Freq"], fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Horizontal separators between datasets
    for sep in [len(ARCHS) - 0.5, 2 * len(ARCHS) - 0.5]:
        ax.axhline(sep, color="white", linewidth=3)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("← PhysGuard Better  |  DFT Better →", fontsize=11)
    # Fix colorbar ticks: show original values
    cbar_ticks = cbar.get_ticks()
    cbar.set_ticklabels([f"{-t:+.0f}%" for t in cbar_ticks])

    ax.set_title("PhysGuard vs. DFT: Relative Error Change (%)\n"
                 "Negative (green) = PhysGuard reduces error",
                 fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "006-scheme-E-winlose.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== MAIN ==========================

def main():
    setup_style()
    print("Collecting data from training logs...")
    data = collect_all_data()
    print(f"  Found {len(data)} experiment entries")

    # Quick summary: show a few key values
    print("\nKey values (rel_low_f at best iteration):")
    for ds in DATASETS:
        print(f"\n  {DATASET_NAMES[ds]}:")
        for ar in ARCHS:
            pre = data.get((ds, ar, "pretrained"), {}).get("rel_low_f", float("nan"))
            dft = data.get((ds, ar, "dft"), {}).get("rel_low_f", float("nan"))
            nsft = data.get((ds, ar, "nsft"), {}).get("rel_low_f", float("nan"))
            delta_dft = (dft - pre) / pre * 100 if pre > 0 else float("nan")
            delta_nsft = (nsft - pre) / pre * 100 if pre > 0 else float("nan")
            print(f"    {ARCH_NAMES[ar]:12s}  Pre={pre:.3f}  DFT={dft:.3f}({delta_dft:+.1f}%)  "
                  f"PhysGuard={nsft:.3f}({delta_nsft:+.1f}%)")

    print("\nGenerating 5 schemes...")

    print("\n  Scheme A: Δ% relative to Pretrained")
    scheme_A_delta_pct(data)

    print("  Scheme B: Frequency band decomposition")
    scheme_B_stacked_bar(data)

    print("  Scheme C: Paired slope chart")
    scheme_C_slope(data)

    print("  Scheme D: Multi-architecture frequency profile")
    scheme_D_freq_profile(data)

    print("  Scheme E: Win/Lose heatmap")
    scheme_E_winlose(data)

    print("\nDone! All 5 schemes saved to figures/006-scheme-*.png")


if __name__ == "__main__":
    main()
