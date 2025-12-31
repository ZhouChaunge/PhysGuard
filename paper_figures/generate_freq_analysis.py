#!/usr/bin/env python3
"""Generate 4 frequency-domain analysis visualizations from training logs.

Figures:
  1. Radar chart  — frequency profile per method (3 subplots, one per dataset)
  2. Violin plot  — distribution of rel_f across (dataset, arch) combos
  3. Slope chart  — frequency band trends with confidence intervals
  4. Scatter plot — low-freq vs high-freq trade-off
"""

import os, re, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
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

COLORS = {
    "Pretrained": "#95a5a6",
    "DFT":        "#27ae60",
    "L2-SP":      "#e74c3c",
    "EWC":        "#3498db",
    "PhysGuard":  "#f39c12",
}

MARKERS = {
    "Pretrained": "o", "DFT": "s", "L2-SP": "^",
    "EWC": "D", "PhysGuard": "*",
}

FREQ_BANDS = ["Low", "Mid", "High"]
REL_KEYS   = ["rel_low_f", "rel_mid_f", "rel_high_f"]

# ========================== LOG PARSING ==========================

def parse_log_path(log_path):
    """Extract (dataset, arch, method, date_str) from training log path."""
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

    # Extract date
    date_str = None
    for p in parts[idx + 3:]:
        if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", p):
            date_str = p
            break

    return dataset, arch, method, date_str


def parse_nsft_line(line):
    """Parse a single nsft-format validation line (comma-separated key: value)."""
    metrics = {}
    chunks = line.split(", ")
    # Longest keys first to avoid partial matches
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


def parse_metrics_from_log(log_path):
    """Return the LAST validation's frequency metrics from a training log."""
    with open(log_path, "r") as f:
        content = f.read()

    # ---- train_gpus.py format: key=value ----
    if "rel_low_f=" in content:
        lines = content.split("\n")
        last_idx = None
        for i, line in enumerate(lines):
            if "rel_low_f=" in line:
                last_idx = i
        if last_idx is None:
            return None
        block = "\n".join(lines[max(0, last_idx - 3) : last_idx + 1])
        metrics = {}
        for key in [
            "low_f", "mid_f", "high_f",
            "rel_low_f", "rel_mid_f", "rel_high_f",
            "rmse", "rel_l2", "f_err", "freq_err", "ke_err",
        ]:
            m = re.search(rf"(?<![a-z_]){key}=([\d.eE+-]+)", block)
            if m:
                metrics[key] = float(m.group(1))
        return metrics if "rel_low_f" in metrics else None

    # ---- train_nullspace.py format: "key: value" ----
    if "rel low f error" in content:
        lines = content.split("\n")
        last_val_line = None
        for line in lines:
            if "rel low f error" in line:
                last_val_line = line
        if last_val_line is None:
            return None
        return parse_nsft_line(last_val_line)

    return None


def collect_all_data():
    """Scan all training logs → dict keyed by (dataset, arch, method)."""
    # Find all training.log files
    logs = []
    for root, dirs, files in os.walk(RESULTS_DIR):
        if "training.log" in files:
            logs.append(os.path.join(root, "training.log"))

    # Group by (dataset, arch, method) and keep latest run
    groups = defaultdict(list)
    for log_path in logs:
        dataset, arch, method, date_str = parse_log_path(log_path)
        if dataset in DATASETS and arch in ARCHS and method in METHOD_ORDER:
            groups[(dataset, arch, method)].append((date_str or "", log_path))

    data = {}
    for key, runs in groups.items():
        runs.sort(key=lambda x: x[0], reverse=True)   # latest first
        metrics = parse_metrics_from_log(runs[0][1])
        if metrics:
            data[key] = metrics

    return data


# ========================== DATA HELPERS ==========================

def build_matrix(data, metric_key):
    """Return array shape (n_datasets, n_archs, n_methods) for a given metric."""
    mat = np.full((len(DATASETS), len(ARCHS), len(METHOD_ORDER)), np.nan)
    for di, ds in enumerate(DATASETS):
        for ai, ar in enumerate(ARCHS):
            for mi, mt in enumerate(METHOD_ORDER):
                rec = data.get((ds, ar, mt))
                if rec and metric_key in rec:
                    mat[di, ai, mi] = rec[metric_key]
    return mat


def method_label(m):
    return METHOD_NAMES[m]


def method_color(m):
    return COLORS[METHOD_NAMES[m]]


def method_marker(m):
    return MARKERS[METHOD_NAMES[m]]


# ========================== FIGURE 1: RADAR CHART ==========================

def fig_radar(data):
    """Radar chart: frequency profile per method, one subplot per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        ax.set_title(DATASET_NAMES[ds], fontsize=14, fontweight="bold", pad=20)

        for mi, mt in enumerate(METHOD_ORDER):
            vals = []
            for band_key in REL_KEYS:
                band_vals = []
                for ai, ar in enumerate(ARCHS):
                    rec = data.get((ds, ar, mt))
                    if rec and band_key in rec:
                        band_vals.append(rec[band_key])
                vals.append(np.mean(band_vals) if band_vals else np.nan)

            vals_closed = vals + vals[:1]
            color = method_color(mt)
            label = method_label(mt)
            ax.plot(angles, vals_closed, "o-", color=color, linewidth=2,
                    label=label, markersize=6)
            ax.fill(angles, vals_closed, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(FREQ_BANDS, fontsize=12)
        ax.tick_params(axis="y", labelsize=9)
        # Annotate: smaller = better
        ax.annotate("← better", xy=(0.5, -0.08), xycoords="axes fraction",
                     ha="center", fontsize=9, color="gray")

    # One legend for all
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Frequency Band Profile (Relative Error, averaged over architectures)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "005-freq-radar.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== FIGURE 2: VIOLIN / BOX PLOT ==========================

def fig_violin(data):
    """Violin plot: distribution of rel_f across (dataset, arch) combos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for bi, (band, key) in enumerate(zip(FREQ_BANDS, REL_KEYS)):
        ax = axes[bi]
        ax.set_title(f"{band}-Frequency Band", fontsize=14, fontweight="bold")

        positions = np.arange(len(METHOD_ORDER))
        all_data = []
        for mi, mt in enumerate(METHOD_ORDER):
            vals = []
            for ds in DATASETS:
                for ar in ARCHS:
                    rec = data.get((ds, ar, mt))
                    if rec and key in rec:
                        vals.append(rec[key])
            all_data.append(vals)

        # Violin
        parts = ax.violinplot(all_data, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False)
        for pi, pc in enumerate(parts["bodies"]):
            color = method_color(METHOD_ORDER[pi])
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(color)

        # Box overlay
        bp = ax.boxplot(all_data, positions=positions, widths=0.3,
                        patch_artist=True, showfliers=False, zorder=3)
        for pi, patch in enumerate(bp["boxes"]):
            color = method_color(METHOD_ORDER[pi])
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor("black")
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color("black")
                item.set_linewidth(1.2)

        # Jittered individual points
        for mi, (mt, vals) in enumerate(zip(METHOD_ORDER, all_data)):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            color = method_color(mt)
            ax.scatter(positions[mi] + jitter, vals, s=25, color=color,
                       edgecolors="white", linewidths=0.5, zorder=4, alpha=0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels([method_label(m) for m in METHOD_ORDER],
                           fontsize=11, rotation=15)
        ax.set_ylabel("Relative Frequency Error" if bi == 0 else "", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)

    fig.suptitle("Distribution of Relative Frequency Error across All Datasets × Architectures",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "005-freq-violin.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== FIGURE 3: SLOPE CHART ==========================

def fig_slope(data):
    """Slope chart: Low → Mid → High trend with confidence band."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.array([0, 1, 2])

    for mi, mt in enumerate(METHOD_ORDER):
        band_values = {b: [] for b in range(3)}
        for ds in DATASETS:
            for ar in ARCHS:
                rec = data.get((ds, ar, mt))
                if rec:
                    for bi, key in enumerate(REL_KEYS):
                        if key in rec:
                            band_values[bi].append(rec[key])

        means = [np.mean(band_values[b]) if band_values[b] else np.nan for b in range(3)]
        stds  = [np.std(band_values[b])  if band_values[b] else 0 for b in range(3)]
        means = np.array(means)
        stds  = np.array(stds)

        color = method_color(mt)
        label = method_label(mt)
        marker = method_marker(mt)

        ax.plot(x_pos, means, f"-{marker}", color=color, linewidth=2.5,
                markersize=10, label=label, zorder=5)
        ax.fill_between(x_pos, means - stds, means + stds,
                         alpha=0.15, color=color, zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(FREQ_BANDS, fontsize=13)
    ax.set_ylabel("Relative Frequency Error (mean ± std)", fontsize=12)
    ax.set_xlabel("Frequency Band", fontsize=12)
    ax.legend(fontsize=11, frameon=True, loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Frequency Band Trend (across all datasets × architectures)",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "005-freq-slope.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== FIGURE 4: SCATTER ==========================

def fig_scatter(data):
    """Scatter: rel_low_f vs rel_high_f, one subplot per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        ax.set_title(DATASET_NAMES[ds], fontsize=14, fontweight="bold")

        for mi, mt in enumerate(METHOD_ORDER):
            xs, ys = [], []
            for ar in ARCHS:
                rec = data.get((ds, ar, mt))
                if rec and "rel_low_f" in rec and "rel_high_f" in rec:
                    xs.append(rec["rel_low_f"])
                    ys.append(rec["rel_high_f"])

            color  = method_color(mt)
            marker = method_marker(mt)
            label  = method_label(mt)
            size   = 120 if mt == "nsft" else 80

            ax.scatter(xs, ys, c=color, marker=marker, s=size,
                       edgecolors="black", linewidths=0.6, label=label,
                       zorder=5, alpha=0.85)

            # Draw mean marker with larger size + black outline
            if xs:
                ax.scatter([np.mean(xs)], [np.mean(ys)], c=color, marker=marker,
                           s=250, edgecolors="black", linewidths=2, zorder=6,
                           alpha=1.0)

        # Diagonal reference line (equal rel error in both bands)
        lims = [ax.get_xlim(), ax.get_ylim()]
        lo = min(lims[0][0], lims[1][0])
        hi = max(lims[0][1], lims[1][1])
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.4, zorder=1)

        ax.set_xlabel("Relative Low-Freq Error", fontsize=12)
        ax.set_ylabel("Relative High-Freq Error" if di == 0 else "", fontsize=12)
        ax.grid(alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Low-Freq vs High-Freq Relative Error Trade-off\n(small markers = individual archs, large = mean)",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "005-freq-scatter.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ========================== MAIN ==========================

def print_coverage(data):
    """Print a coverage + data table for verification."""
    print("\n" + "=" * 90)
    print("DATA COVERAGE TABLE")
    print("=" * 90)
    for ds in DATASETS:
        print(f"\n{'─' * 90}")
        print(f"  {DATASET_NAMES[ds]}")
        print(f"{'─' * 90}")
        header = f"  {'Arch':<12}" + "".join(f"{method_label(m):>14}" for m in METHOD_ORDER)
        print(header)
        print(f"  {'':─<12}" + "─" * (14 * len(METHOD_ORDER)))
        for ar in ARCHS:
            row = f"  {ARCH_NAMES[ar]:<12}"
            for mt in METHOD_ORDER:
                rec = data.get((ds, ar, mt))
                if rec and "rel_low_f" in rec:
                    row += f"{rec['rel_low_f']:>14.5f}"
                else:
                    row += f"{'MISSING':>14}"
            print(row)

    # Summary
    total = len(DATASETS) * len(ARCHS) * len(METHOD_ORDER)
    found = sum(1 for ds in DATASETS for ar in ARCHS for mt in METHOD_ORDER
                if (ds, ar, mt) in data and "rel_low_f" in data[(ds, ar, mt)])
    print(f"\n  Coverage: {found}/{total} ({100*found/total:.0f}%)")
    print("=" * 90)


def main():
    print("Collecting data from training logs...")
    data = collect_all_data()
    print(f"  Found {len(data)} experiment entries")
    print_coverage(data)

    print("\nGenerating figures...")
    fig_radar(data)
    fig_violin(data)
    fig_slope(data)
    fig_scatter(data)
    print("\nDone!")


if __name__ == "__main__":
    main()
