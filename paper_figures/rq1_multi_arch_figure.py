#!/usr/bin/env python3
"""
Multi-Architecture FIM vs Random — 优化图表（直接用 cache 数据绘图）
=====================================================================
Run:
  python \
      ./scripts/rq1_multi_arch_figure.py
"""

import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = "./figures"
CACHE_DIR  = "/tmp/rq1_multi_arch_cache"
FNO_NPZ    = f"{OUTPUT_DIR}/rq1_random_baseline_data.npz"

K = 20

# ─── 颜色与标记 ───────────────────────────────────────────────────
ARCH_COLOR  = {"FNO": "#4C72B0", "CNO": "#DD8452",
               "DeepONet": "#AAAAAA", "Transolver": "#8172B2"}
ARCH_MARKER = {"FNO": "o", "CNO": "s", "DeepONet": "^", "Transolver": "P"}
ARCH_LABEL  = {"FNO": "FNO", "CNO": "CNO",
               "DeepONet": "DeepONet†", "Transolver": "Transolver"}

# ─── 加载数据 ─────────────────────────────────────────────────────
raw = {
    "FNO"       : np.load(FNO_NPZ),
    "CNO"       : np.load(f"{CACHE_DIR}/CNO.npz"),
    "DeepONet"  : np.load(f"{CACHE_DIR}/DeepONet.npz"),
    "Transolver": np.load(f"{CACHE_DIR}/Transolver.npz"),
}

results = {}
for arch, d in raw.items():
    results[arch] = (d["fl_top"][:K], d["fl_random"][:K])

# ─── 统计 ─────────────────────────────────────────────────────────
stats = {}
for arch, (fl_top, fl_rand) in results.items():
    _, pval = mannwhitneyu(fl_top, fl_rand, alternative='greater')
    delta   = fl_top.mean() - fl_rand.mean()
    stats[arch] = dict(mu_fim=fl_top.mean(), mu_rand=fl_rand.mean(),
                       se_fim=fl_top.std()/np.sqrt(K),
                       se_rand=fl_rand.std()/np.sqrt(K),
                       delta=delta, pval=pval)
    stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else
            ("*" if pval < 0.05 else "n.s."))
    print(f"{arch:<12} FIM={fl_top.mean():.4f}±{fl_top.std():.4f}  "
          f"Rand={fl_rand.mean():.4f}±{fl_rand.std():.4f}  "
          f"Δ={delta:+.4f}  p={pval:.2e} {stars}")

# ─── 图表设置 ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.linewidth"   : 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.grid"        : True,
    "grid.alpha"       : 0.25,
    "grid.linestyle"   : ":",
    "grid.linewidth"   : 0.6,
})

archs = ["FNO", "CNO", "DeepONet", "Transolver"]

fig = plt.figure(figsize=(13, 5.0))
gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.65, 1.0], wspace=0.38)
ax_slope = fig.add_subplot(gs[0])
ax_delta = fig.add_subplot(gs[1])

# ════════════════════════════════════════════════════════════════
# Panel (a): Slope / Dumbbell chart
# ════════════════════════════════════════════════════════════════
X_RAND = 0.0
X_FIM  = 1.0

for arch in archs:
    s     = stats[arch]
    color = ARCH_COLOR[arch]
    mark  = ARCH_MARKER[arch]
    label = ARCH_LABEL[arch]
    is_outlier = (arch == "DeepONet")

    lw      = 1.5 if is_outlier else 2.2
    alpha   = 0.45 if is_outlier else 0.78
    ls      = "--" if is_outlier else "-"
    ms      = 9   if is_outlier else 11
    mew     = 1.8 if is_outlier else 2.2

    # 连接线
    ax_slope.plot([X_RAND, X_FIM], [s["mu_rand"], s["mu_fim"]],
                  color=color, lw=lw, alpha=alpha, ls=ls, zorder=2)

    # Random 端点（空心符号）
    ax_slope.errorbar(X_RAND, s["mu_rand"], yerr=2*s["se_rand"],
                      fmt=mark, color=color, markersize=ms, capsize=4,
                      lw=1.5, zorder=4, alpha=0.85 if not is_outlier else 0.45,
                      markerfacecolor='white', markeredgewidth=mew)

    # FIM 端点（实心符号）
    ax_slope.errorbar(X_FIM, s["mu_fim"], yerr=2*s["se_fim"],
                      fmt=mark, color=color, markersize=ms, capsize=4,
                      lw=1.5, zorder=4, alpha=0.85 if not is_outlier else 0.45,
                      label=label)

    # FIM 侧数值标注
    offset = 0.030
    va     = 'center'
    ax_slope.text(X_FIM + offset, s["mu_fim"], f"{s['mu_fim']:.3f}",
                  va=va, ha='left', fontsize=8.5, color=color,
                  fontweight='bold' if not is_outlier else 'normal',
                  alpha=1.0 if not is_outlier else 0.5)

# 坐标轴装饰
ax_slope.set_xticks([X_RAND, X_FIM])
ax_slope.set_xticklabels(["Random\ndirections",
                           f"FIM top-{K}\ndirections"], fontsize=10.5)
ax_slope.set_xlim(-0.18, 1.30)
ax_slope.set_ylim(-0.02, 1.10)
ax_slope.set_ylabel(r"Low-freq. energy fraction  $f_{\mathrm{low}}$", fontsize=11)
ax_slope.set_title(r"(a)  Slope chart: $f_{\mathrm{low}}$(FIM) vs $f_{\mathrm{low}}$(random)"
                   "\n(each arc = one architecture, open = random, filled = FIM)",
                   fontsize=9.5, pad=6)

# 参考网格线
for y in [0.0, 0.25, 0.5, 0.75, 1.0]:
    ax_slope.axhline(y, color='#dddddd', lw=0.8, zorder=1)

legend = ax_slope.legend(loc='upper left', fontsize=9.5,
                         framealpha=0.92, edgecolor='#cccccc',
                         handlelength=0.9, borderpad=0.7,
                         labelspacing=0.4)
# DeepONet 图例项稍透明
for i, (arch, handle) in enumerate(zip(archs, legend.legend_handles)):
    if arch == "DeepONet":
        handle.set_alpha(0.45)

# ════════════════════════════════════════════════════════════════
# Panel (b): Δ 条形图（正确的标签顺序）
# ════════════════════════════════════════════════════════════════
n     = len(archs)
# 让 FNO 在顶部（y 值最大）时，archs[0]=FNO → y_pos[0] = n-1 = 3
y_pos = np.arange(n)[::-1]   # [3, 2, 1, 0]

deltas = [stats[a]["delta"] for a in archs]
pvals  = [stats[a]["pval"]  for a in archs]
colors = [ARCH_COLOR[a]      for a in archs]
alphas = [0.40 if a == "DeepONet" else 0.82 for a in archs]

for i, (arch, yp, delta, pval, col, al) in enumerate(
        zip(archs, y_pos, deltas, pvals, colors, alphas)):

    bar = ax_delta.barh(yp, delta, color=col, alpha=al, height=0.52,
                        edgecolor='white', linewidth=0.5)

    stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else
            ("*"   if pval < 0.05 else "n.s."))
    annot = f"  {delta:.3f} {stars}"
    fw    = 'normal' if arch == "DeepONet" else 'bold'
    ax_delta.text(delta + 0.004, yp, annot,
                  va='center', ha='left', fontsize=8.5,
                  color=col, fontweight=fw, alpha=al + 0.15)

# ─── 关键：标签顺序与条形顺序一致 ────────────────────────────────
# y_pos = [3,2,1,0] 与 archs = [FNO, CNO, DeepONet, DPOT] 对应
# set_yticklabels 按 y_pos 顺序（3→2→1→0）分配标签
ax_delta.set_yticks(y_pos)                      # [3, 2, 1, 0]
ax_delta.set_yticklabels([ARCH_LABEL[a] for a in archs],
                         fontsize=11)           # labels assigned in order

ax_delta.set_xlabel(r"$\Delta = \bar{f}^{\,\mathrm{FIM}}_{\mathrm{low}}"
                    r" - \bar{f}^{\,\mathrm{rand}}_{\mathrm{low}}$",
                    fontsize=11)
ax_delta.set_title(r"(b)  Low-freq. alignment gap  $\Delta$"
                   "\n(higher = more physically aligned)",
                   fontsize=9.5, pad=6)
ax_delta.set_xlim(0, max(deltas) * 1.45)
ax_delta.axvline(0, color='#888888', lw=0.9, zorder=2)

# 去掉无用的 y 方向网格
ax_delta.grid(axis='x', ls=':', alpha=0.3, lw=0.6)
ax_delta.grid(axis='y', visible=False)

# ─── 足注（DeepONet 说明）────────────────────────────────────────
fig.text(0.5, -0.04,
         "† DeepONet: the trunk MLP maps raw spatial coordinates (t x y) to smooth basis functions. "
         "Both FIM and random perturbations of trunk weights produce smooth spatial changes, "
         "so $f_{\\mathrm{low}}\\approx 1$ for all directions and $\\Delta\\approx 0$. "
         "This reflects DeepONet’s inherent low-frequency spatial inductive bias.",
         ha='center', fontsize=8, color='#555555', style='italic',
         wrap=True)

# ─── 总标题 ───────────────────────────────────────────────────────
fig.suptitle(
    "FIM principal subspace preferentially aligns with low-frequency physical modes\n"
    r"(Cylinder dataset · $\varepsilon = 10^{-3}$ · $K = 20$ directions · 95% CI shown)",
    fontsize=11.5, y=1.03)

fig.tight_layout(rect=[0, 0.04, 1, 1])

for ext in ["pdf", "png"]:
    path = f"{OUTPUT_DIR}/rq1_multi_arch_vs_random.{ext}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
plt.close(fig)
