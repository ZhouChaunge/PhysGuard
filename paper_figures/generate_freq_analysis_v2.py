#!/usr/bin/env python3
"""Generate improved frequency-domain visualizations with architecture × method differentiation.

Figures:
  A. Heatmap matrix (compact overview)
  B. Cleveland dot plot (precise comparison)
  C. Faceted grouped bar chart (universal readability)
  D. Improved violin+swarm with identity annotations
"""

import os, re, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
DATASET_SHORT = {
    "001-cylinder":          "Cyl",
    "002-control_cylinder":  "CtrlCyl",
    "003-combustion":        "Comb",
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

MARKERS_DS = {
    "Cylinder Flow": "o", "Controlled Cylinder": "s", "Combustion": "D",
}

FREQ_BANDS = ["Low", "Mid", "High"]
REL_KEYS   = ["rel_low_f", "rel_mid_f", "rel_high_f"]
ABS_KEYS   = ["low_f", "mid_f", "high_f"]

# ========================== LOG PARSING (same as before) ==========================

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
    """Extract the best iteration number from the log."""
    # train_gpus.py: "训练完成，best iteration=3760，耗时"
    m = re.search(r"best iteration[=:]\s*(\d+)", content, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # train_nullspace.py: "Best iteration: 400, Best val RMSE:"
    m = re.search(r"Best iteration:\s*(\d+)", content)
    if m:
        return int(m.group(1))
    return None


def parse_gpus_block(lines, target_iter):
    """Parse train_gpus.py format: find Iteration <target_iter> and extract the
    subsequent validation block (3 lines: header, ke/f, rel_f)."""
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
    """Parse train_nullspace.py format: find Iteration <target_iter> and extract
    the subsequent 'Validation results:' line."""
    for i, line in enumerate(lines):
        if f"Iteration {target_iter}," in line:
            # The validation line is typically 1-2 lines after
            for j in range(i, min(i + 5, len(lines))):
                if "rel low f error" in lines[j]:
                    return parse_nsft_line(lines[j])
    return None


def parse_metrics_from_log(log_path):
    """Extract frequency metrics at the BEST iteration from a training log."""
    with open(log_path, "r") as f:
        content = f.read()

    lines = content.split("\n")
    best_iter = find_best_iteration(content)

    # ---- train_gpus.py format: key=value ----
    if "rel_low_f=" in content:
        if best_iter is not None:
            metrics = parse_gpus_block(lines, best_iter)
            if metrics:
                return metrics
        # Fallback: last validation line
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

    # ---- train_nullspace.py format: "key: value" ----
    if "rel low f error" in content:
        if best_iter is not None:
            metrics = parse_nsft_block(lines, best_iter)
            if metrics:
                return metrics
        # Fallback: last validation line
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
    """Set publication-quality defaults."""
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


# ========================== FIGURE A: HEATMAP ==========================

def fig_heatmap(data):
    """Heatmap: rows=arch, cols=method, 3 subplots (datasets), focus on rel_low_f."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        mat = np.full((len(ARCHS), len(METHOD_ORDER)), np.nan)
        for ai, ar in enumerate(ARCHS):
            for mi, mt in enumerate(METHOD_ORDER):
                rec = data.get((ds, ar, mt))
                if rec and "rel_low_f" in rec:
                    mat[ai, mi] = rec["rel_low_f"]

        # Color: lower = better = green, higher = worse = red
        vmin = np.nanmin(mat) * 0.95
        vmax = np.nanmax(mat) * 1.02
        im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)

        # Annotate each cell
        for ai in range(len(ARCHS)):
            for mi in range(len(METHOD_ORDER)):
                val = mat[ai, mi]
                if not np.isnan(val):
                    # Bold the best (lowest) in each row
                    row_min = np.nanmin(mat[ai])
                    weight = "bold" if abs(val - row_min) < 1e-5 else "normal"
                    text_color = "white" if val > (vmin + vmax) / 2 else "black"
                    ax.text(mi, ai, f"{val:.3f}", ha="center", va="center",
                            fontsize=9.5, fontweight=weight, color=text_color)

        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_NAMES[m] for m in METHOD_ORDER], rotation=30, ha="right")
        ax.set_yticks(range(len(ARCHS)))
        ax.set_yticklabels([ARCH_NAMES[a] for a in ARCHS])
        ax.set_title(DATASET_NAMES[ds], fontweight="bold", fontsize=13)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        if di == 2:
            cbar.set_label("Relative Low-Freq Error ↓", fontsize=10)

    fig.suptitle("Low-Frequency Preservation (Relative Error)\nBold = best per architecture, Green = better",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "005-freq-heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== FIGURE B: DOT PLOT ==========================

def fig_dotplot(data):
    """Cleveland dot plot: y=arch, x=metric, different markers/colors per method."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    y_positions = np.arange(len(ARCHS))
    offsets = np.linspace(-0.3, 0.3, len(METHOD_ORDER))  # vertical jitter per method

    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        ax.set_title(DATASET_NAMES[ds], fontweight="bold", fontsize=13)

        for mi, mt in enumerate(METHOD_ORDER):
            xs = []
            ys = []
            for ai, ar in enumerate(ARCHS):
                rec = data.get((ds, ar, mt))
                if rec and "rel_low_f" in rec:
                    xs.append(rec["rel_low_f"])
                    ys.append(y_positions[ai] + offsets[mi])

            color = COLORS[METHOD_NAMES[mt]]
            label = METHOD_NAMES[mt]
            ax.scatter(xs, ys, c=color, s=90, marker="o",
                       edgecolors="white", linewidths=0.8,
                       label=label if di == 0 else None, zorder=5)

        # Connect same arch across methods with thin lines
        for ai, ar in enumerate(ARCHS):
            vals = []
            for mt in METHOD_ORDER:
                rec = data.get((ds, ar, mt))
                if rec and "rel_low_f" in rec:
                    vals.append((rec["rel_low_f"], y_positions[ai] + offsets[METHOD_ORDER.index(mt)]))
            if len(vals) > 1:
                xs_line = [v[0] for v in vals]
                ys_line = [v[1] for v in vals]
                ax.plot(xs_line, ys_line, '-', color='#cccccc', linewidth=0.8, zorder=1)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([ARCH_NAMES[a] for a in ARCHS], fontsize=11)
        ax.set_xlabel("Relative Low-Freq Error ↓", fontsize=11)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.invert_yaxis()

    fig.legend(loc="lower center", ncol=5, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.05),
               markerscale=1.2)
    fig.suptitle("Low-Frequency Error by Architecture and Method",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "005-freq-dotplot.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== FIGURE C: GROUPED BAR ==========================

def fig_grouped_bar(data):
    """Grouped bar chart: faceted by dataset, grouped by arch, colored by method.
    Focus on rel_low_f only for clarity."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

    n_methods = len(METHOD_ORDER)
    bar_width = 0.15
    x_base = np.arange(len(ARCHS))

    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        ax.set_title(DATASET_NAMES[ds], fontweight="bold", fontsize=13)

        for mi, mt in enumerate(METHOD_ORDER):
            vals = []
            for ar in ARCHS:
                rec = data.get((ds, ar, mt))
                vals.append(rec["rel_low_f"] if rec and "rel_low_f" in rec else 0)

            offset = (mi - n_methods / 2 + 0.5) * bar_width
            color = COLORS[METHOD_NAMES[mt]]
            bars = ax.bar(x_base + offset, vals, bar_width * 0.9,
                          color=color, edgecolor="white", linewidth=0.5,
                          label=METHOD_NAMES[mt] if di == 0 else None,
                          zorder=3)

            # Value labels on bars
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                            rotation=90, color="#333333")

        ax.set_xticks(x_base)
        ax.set_xticklabels([ARCH_NAMES[a] for a in ARCHS], fontsize=11)
        ax.set_ylabel("Relative Low-Freq Error ↓" if di == 0 else "", fontsize=11)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_xlim(-0.6, len(ARCHS) - 0.3)

    fig.legend(loc="lower center", ncol=5, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Low-Frequency Error Comparison across Architectures",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "005-freq-bars.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== FIGURE D: IMPROVED VIOLIN + SWARM ==========================

def fig_violin_improved(data):
    """Improved violin: one subplot per freq band.
    Each method's violin + individual dots annotated by arch shape and dataset color."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), sharey=False)

    ds_colors = {
        "Cylinder Flow":        "#2c3e50",
        "Controlled Cylinder":  "#16a085",
        "Combustion":           "#c0392b",
    }

    for bi, (band, key) in enumerate(zip(FREQ_BANDS, REL_KEYS)):
        ax = axes[bi]
        ax.set_title(f"{band}-Frequency Band", fontweight="bold", fontsize=13)

        positions = np.arange(len(METHOD_ORDER))

        # Collect data per method
        all_data = []
        all_meta = []  # list of list of (ds_name, arch_name)
        for mi, mt in enumerate(METHOD_ORDER):
            vals = []
            meta = []
            for ds in DATASETS:
                for ar in ARCHS:
                    rec = data.get((ds, ar, mt))
                    if rec and key in rec:
                        vals.append(rec[key])
                        meta.append((DATASET_NAMES[ds], ARCH_NAMES[ar]))
            all_data.append(vals)
            all_meta.append(meta)

        # Violin background
        parts = ax.violinplot(all_data, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False, widths=0.7)
        for pi, pc in enumerate(parts["bodies"]):
            color = COLORS[METHOD_NAMES[METHOD_ORDER[pi]]]
            pc.set_facecolor(color)
            pc.set_alpha(0.15)
            pc.set_edgecolor(color)
            pc.set_linewidth(1)

        # Mean line
        for mi, (mt, vals) in enumerate(zip(METHOD_ORDER, all_data)):
            if vals:
                mean_val = np.mean(vals)
                ax.plot([positions[mi] - 0.2, positions[mi] + 0.2],
                        [mean_val, mean_val], '-', color=COLORS[METHOD_NAMES[mt]],
                        linewidth=2.5, zorder=6)

        # Swarm dots: shape = arch, color = dataset
        rng = np.random.default_rng(42)
        for mi, (mt, vals, meta) in enumerate(zip(METHOD_ORDER, all_data, all_meta)):
            for vi, (val, (ds_name, ar_name)) in enumerate(zip(vals, meta)):
                jitter = rng.uniform(-0.18, 0.18)
                marker = MARKERS_ARCH[ar_name]
                color = ds_colors[ds_name]
                ax.scatter(positions[mi] + jitter, val, s=55, marker=marker,
                           c=color, edgecolors="white", linewidths=0.5,
                           zorder=5, alpha=0.85)

        ax.set_xticks(positions)
        ax.set_xticklabels([METHOD_NAMES[m] for m in METHOD_ORDER],
                           fontsize=10.5, rotation=15)
        ax.set_ylabel("Relative Frequency Error ↓" if bi == 0 else "", fontsize=11)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)

    # Build compound legend
    # Method (color bar at mean)
    method_handles = [Line2D([0], [0], color=COLORS[METHOD_NAMES[mt]], linewidth=3,
                              label=METHOD_NAMES[mt]) for mt in METHOD_ORDER]
    # Dataset (color of dots)
    ds_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=8, label=n) for n, c in ds_colors.items()]
    # Architecture (marker shape)
    arch_handles = [Line2D([0], [0], marker=MARKERS_ARCH[ARCH_NAMES[a]], color='w',
                            markerfacecolor='#555555', markersize=8,
                            label=ARCH_NAMES[a]) for a in ARCHS]

    leg1 = fig.legend(handles=method_handles, title="Method", loc="lower left",
                      ncol=5, fontsize=9.5, title_fontsize=10,
                      bbox_to_anchor=(0.02, -0.12), frameon=True)
    leg2 = fig.legend(handles=ds_handles, title="Dataset", loc="lower center",
                      ncol=3, fontsize=9.5, title_fontsize=10,
                      bbox_to_anchor=(0.5, -0.12), frameon=True)
    leg3 = fig.legend(handles=arch_handles, title="Architecture", loc="lower right",
                      ncol=5, fontsize=9.5, title_fontsize=10,
                      bbox_to_anchor=(0.98, -0.12), frameon=True)
    fig.add_artist(leg1)
    fig.add_artist(leg2)

    fig.suptitle("Frequency Error Distribution with Architecture & Dataset Identity",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "005-freq-violin-improved.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== BONUS: MULTI-BAND HEATMAP ==========================

def fig_heatmap_3band(data):
    """3-band heatmap: rows = (dataset, arch), cols = method, 3 sub-columns per method for Low/Mid/High."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))

    for bi, (band, key) in enumerate(zip(FREQ_BANDS, REL_KEYS)):
        ax = axes[bi]
        ax.set_title(f"{band}-Frequency Band", fontweight="bold", fontsize=13, pad=10)

        # Build matrix: rows = datasets × archs flattened, cols = methods
        n_rows = len(DATASETS) * len(ARCHS)
        mat = np.full((n_rows, len(METHOD_ORDER)), np.nan)
        row_labels = []
        for di, ds in enumerate(DATASETS):
            for ai, ar in enumerate(ARCHS):
                ri = di * len(ARCHS) + ai
                row_labels.append(f"{DATASET_SHORT[ds]} / {ARCH_NAMES[ar]}")
                for mi, mt in enumerate(METHOD_ORDER):
                    rec = data.get((ds, ar, mt))
                    if rec and key in rec:
                        mat[ri, mi] = rec[key]

        vmin = np.nanmin(mat) * 0.95
        vmax = np.nanmax(mat) * 1.02
        im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)

        for ri in range(n_rows):
            for mi in range(len(METHOD_ORDER)):
                val = mat[ri, mi]
                if not np.isnan(val):
                    row_min = np.nanmin(mat[ri])
                    weight = "bold" if abs(val - row_min) < 1e-5 else "normal"
                    text_color = "white" if val > (vmin + vmax) * 0.55 else "black"
                    ax.text(mi, ri, f"{val:.3f}", ha="center", va="center",
                            fontsize=7.5, fontweight=weight, color=text_color)

        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_NAMES[m] for m in METHOD_ORDER], fontsize=10)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=9)

        # Horizontal lines to separate datasets
        for sep in [len(ARCHS) - 0.5, 2 * len(ARCHS) - 0.5]:
            ax.axhline(sep, color="white", linewidth=2)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Frequency Error Overview: All Datasets × Architectures × Methods\nBold = best per row, Green = better",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "005-freq-heatmap-3band.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== MAIN ==========================

def main():
    setup_style()
    print("Collecting data from training logs...")
    data = collect_all_data()
    print(f"  Found {len(data)} experiment entries")

    # Verification: print table with BOTH absolute and relative Low-f
    print("\n" + "=" * 110)
    print("VERIFICATION: low_f (absolute, matches paper Table 1) and rel_low_f (relative, used in figures)")
    print("=" * 110)

    # Paper Table 1 reference values for low_f (Cylinder Flow only)
    paper_low_f = {
        ("001-cylinder", "fno", "pretrained"): 0.01829,
        ("001-cylinder", "fno", "dft"):        0.01651,
        ("001-cylinder", "fno", "l2ft"):       0.01739,
        ("001-cylinder", "fno", "ewcft"):      0.01596,
        ("001-cylinder", "fno", "nsft"):       0.01131,
        ("001-cylinder", "cno", "pretrained"): 0.01379,
        ("001-cylinder", "cno", "dft"):        0.01585,
        ("001-cylinder", "cno", "l2ft"):       0.01506,
        ("001-cylinder", "cno", "ewcft"):      0.01355,
        ("001-cylinder", "cno", "nsft"):       0.01148,
        ("001-cylinder", "deeponet", "pretrained"): 0.02793,
        ("001-cylinder", "deeponet", "dft"):        0.02527,
        ("001-cylinder", "deeponet", "l2ft"):       0.02627,
        ("001-cylinder", "deeponet", "ewcft"):      0.02599,
        ("001-cylinder", "deeponet", "nsft"):       0.01957,
        ("001-cylinder", "dpot", "pretrained"): 0.01202,
        ("001-cylinder", "dpot", "dft"):        0.01026,
        ("001-cylinder", "dpot", "l2ft"):       0.01075,
        ("001-cylinder", "dpot", "ewcft"):      0.00989,
        ("001-cylinder", "dpot", "nsft"):       0.00998,
    }

    for ds in DATASETS:
        print(f"\n  {DATASET_NAMES[ds]}")
        header = f"  {'Arch':<12}" + "".join(f"{'low_f':>10}{'rel':>8}{'paper':>8}" if mi == 0
                                              else f"{'low_f':>10}{'rel':>8}{'paper':>8}"
                                              for mi, m in enumerate(METHOD_ORDER))
        # Simplified header
        hdr = f"  {'Arch':<12}"
        for m in METHOD_ORDER:
            hdr += f" | {METHOD_NAMES[m]:^24}"
        print(hdr)
        sub = f"  {'':─<12}"
        for m in METHOD_ORDER:
            sub += f" | {'low_f':>8}{'rel_low_f':>10}{'paper':>8}"
        print(sub)

        for ar in ARCHS:
            row = f"  {ARCH_NAMES[ar]:<12}"
            for mt in METHOD_ORDER:
                rec = data.get((ds, ar, mt))
                pval = paper_low_f.get((ds, ar, mt))
                if rec:
                    lf = rec.get('low_f', float('nan'))
                    rlf = rec.get('rel_low_f', float('nan'))
                    pstr = f"{pval:.5f}" if pval else "    -   "
                    match = "✅" if pval and abs(lf - pval) < 0.00005 else "❌" if pval else "  "
                    row += f" | {lf:>8.5f}{rlf:>10.5f}{pstr:>8}{match}"
                else:
                    row += f" | {'MISS':>8}{'MISS':>10}{'':>8}  "
            print(row)

    print("\n" + "=" * 110)

    print("\nGenerating improved figures...")
    fig_heatmap(data)
    fig_dotplot(data)
    fig_grouped_bar(data)
    fig_violin_improved(data)
    fig_heatmap_3band(data)
    print("\nDone! All 5 figures saved.")


if __name__ == "__main__":
    main()
