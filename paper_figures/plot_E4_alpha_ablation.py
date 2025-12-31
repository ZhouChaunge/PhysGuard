"""
E4 Alpha Ablation Curves
Plots RMSE and Low-f fRMSE training curves for DeepONet + Cylinder Flow
across alpha ∈ {0.3, 0.5, 0.7, 1.0} and DFT baseline.
"""

import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─────────────────────── 配置 ────────────────────────────────────────────────
BASE = "./results/001-cylinder/deeponet"
OUT_DIR = "./figures/ablation"
os.makedirs(OUT_DIR, exist_ok=True)

EXPERIMENTS = {
    "DFT (α=0)":         os.path.join(BASE, "deeponet_cylinder_dft/2026-03-13_22-22-57/training.log"),
    "α=0.3":             os.path.join(BASE, "deeponet_cylinder_ablation_alpha0.3_nsft/2026-04-10_14-18-00/training.log"),
    "α=0.5":             os.path.join(BASE, "deeponet_cylinder_ablation_alpha0.5_nsft/2026-04-10_14-18-22/training.log"),
    "α=0.7":             os.path.join(BASE, "deeponet_cylinder_ablation_alpha0.7_nsft/2026-04-10_14-18-43/training.log"),
    "PhysGuard (α=1.0)": os.path.join(BASE, "deeponet_cylinder_nsft/2026-03-13_00-38-45/training.log"),
}

# 颜色方案
COLORS = {
    "DFT (α=0)":         "#999999",
    "α=0.3":             "#f4a261",
    "α=0.5":             "#e76f51",
    "α=0.7":             "#2a9d8f",
    "PhysGuard (α=1.0)": "#264653",
}
LINESTYLES = {
    "DFT (α=0)":         "--",
    "α=0.3":             "-",
    "α=0.5":             "-",
    "α=0.7":             "-",
    "PhysGuard (α=1.0)": "-",
}

# ─────────────────────── 解析 log ────────────────────────────────────────────
RE_ITER = re.compile(r"Iteration\s+(\d+)")
RE_METRICS = re.compile(
    r"rmse:\s*([\d.]+).*?low f error:\s*([\d.]+)",
    re.DOTALL
)

def parse_log(path):
    """Returns (steps, rmse_list, low_f_list)."""
    steps, rmse_list, low_f_list = [], [], []

    with open(path) as f:
        lines = f.readlines()

    current_iter = None
    for i, line in enumerate(lines):
        m = RE_ITER.search(line)
        if m:
            current_iter = int(m.group(1))

        if "Validation results:" in line and current_iter is not None:
            # metrics are on the very next line
            if i + 1 < len(lines):
                metric_line = lines[i + 1]
                mm = RE_METRICS.search(metric_line)
                if mm:
                    steps.append(current_iter)
                    rmse_list.append(float(mm.group(1)))
                    low_f_list.append(float(mm.group(2)))

    return np.array(steps), np.array(rmse_list), np.array(low_f_list)


# ─────────────────────── 收集数据 ────────────────────────────────────────────
data = {}
best_table = {}  # label -> (best_rmse, best_low_f)

for label, log_path in EXPERIMENTS.items():
    if not os.path.exists(log_path):
        print(f"[WARN] 找不到 log: {log_path}")
        continue
    steps, rmse, low_f = parse_log(log_path)
    data[label] = (steps, rmse, low_f)
    if len(rmse) > 0:
        best_table[label] = (rmse.min(), low_f[rmse.argmin()])
    print(f"{label}: {len(steps)} steps, best RMSE={rmse.min():.5f}, low_f@best={low_f[rmse.argmin()]:.5f}")

# ─────────────────────── 平滑 ────────────────────────────────────────────────
def smooth(arr, w=5):
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")

# ─────────────────────── 绘图 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for label, (steps, rmse, low_f) in data.items():
    color = COLORS[label]
    ls = LINESTYLES[label]
    lw = 2.0 if label != "DFT (α=0)" else 1.5
    kw = dict(color=color, linestyle=ls, linewidth=lw, label=label)

    axes[0].plot(steps, smooth(rmse),  **kw)
    axes[1].plot(steps, smooth(low_f), **kw)

for ax, ylabel, title in zip(
    axes,
    ["RMSE ↓", "Low-$f$ fRMSE ↓"],
    ["(a) RMSE vs. Training Steps", "(b) Low-$f$ fRMSE vs. Training Steps"],
):
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

fig.suptitle("E4: α Ablation — DeepONet on Cylinder Flow (τ=0.90)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_fig = os.path.join(OUT_DIR, "E4_alpha_ablation_curves.pdf")
fig.savefig(out_fig, bbox_inches="tight", dpi=150)
out_png = out_fig.replace(".pdf", ".png")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"\n图像已保存: {out_fig}")
print(f"图像已保存: {out_png}")

# ─────────────────────── 条形图：最终指标对比 ─────────────────────────────
order = ["DFT (α=0)", "α=0.3", "α=0.5", "α=0.7", "PhysGuard (α=1.0)"]
alpha_vals = [0.0, 0.3, 0.5, 0.7, 1.0]
bar_labels = ["DFT\n(α=0)", "α=0.3", "α=0.5", "α=0.7", "PhysGuard\n(α=1.0)"]

best_rmse  = [best_table[l][0] for l in order if l in best_table]
best_lowf  = [best_table[l][1] for l in order if l in best_table]
bar_colors = [COLORS[l] for l in order if l in best_table]
bar_labels_filtered = [bar_labels[i] for i, l in enumerate(order) if l in best_table]

fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4))
x = np.arange(len(best_rmse))
w = 0.55

for ax, vals, ylabel, title in zip(
    axes2,
    [best_rmse, best_lowf],
    ["RMSE ↓", "Low-$f$ fRMSE ↓"],
    ["(a) Best RMSE", "(b) Best Low-$f$ fRMSE"],
):
    bars = ax.bar(x, vals, width=w, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels_filtered, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_ylim(min(vals) * 0.90, max(vals) * 1.06)
    # annotate values
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

fig2.suptitle("E4: α Ablation — Best Val Metrics (DeepONet, Cylinder Flow)", fontsize=11, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
out_bar = os.path.join(OUT_DIR, "E4_alpha_ablation_bar.png")
fig2.savefig(out_bar, bbox_inches="tight", dpi=150)
print(f"条形图已保存: {out_bar}")

# ─────────────────────── 汇总表 ──────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"{'Method':<22}  {'Best RMSE':>10}  {'Low-f fRMSE':>12}")
print("-" * 55)
order = ["DFT (α=0)", "α=0.3", "α=0.5", "α=0.7", "PhysGuard (α=1.0)"]
for label in order:
    if label in best_table:
        r, lf = best_table[label]
        print(f"{label:<22}  {r:>10.5f}  {lf:>12.5f}")
print("=" * 55)

# 保存 CSV
csv_path = os.path.join(OUT_DIR, "E4_alpha_ablation_summary.csv")
with open(csv_path, "w") as f:
    f.write("alpha,method,best_rmse,low_f_frmse\n")
    alpha_map = {
        "DFT (α=0)": "0.0",
        "α=0.3": "0.3",
        "α=0.5": "0.5",
        "α=0.7": "0.7",
        "PhysGuard (α=1.0)": "1.0",
    }
    for label in order:
        if label in best_table:
            r, lf = best_table[label]
            f.write(f"{alpha_map[label]},{label},{r:.5f},{lf:.5f}\n")
print(f"CSV 已保存: {csv_path}")
