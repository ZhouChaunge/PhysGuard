#!/usr/bin/env python3
"""
FIM Top-20 vs Random — 简洁双面板图
=====================================
直接从已保存的 rq1_random_baseline_data.npz 读取数据，
只展示 FIM top-20 与随机方向的对比（去掉 bottom 组）。

Run:
    python \
        ./scripts/rq1_top20_vs_random.py
"""

import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── 配置 ─────────────────────────────────────────────────────────
OUTPUT_DIR = "./figures"
DATA_PATH  = f"{OUTPUT_DIR}/rq1_random_baseline_data.npz"
K_SHOW     = 20   # 只取前 20 个 FIM 方向

# ─── 加载数据 ─────────────────────────────────────────────────────
data      = np.load(DATA_PATH)
fl_top    = data["fl_top"][:K_SHOW]      # shape [20]
fl_random = data["fl_random"][:K_SHOW]   # shape [20]

# ─── 统计检验 ─────────────────────────────────────────────────────
stat, pval = mannwhitneyu(fl_top, fl_random, alternative='greater')
delta      = fl_top.mean() - fl_random.mean()
print(f"FIM top-{K_SHOW}  mean={fl_top.mean():.4f} ± {fl_top.std():.4f}")
print(f"Random      mean={fl_random.mean():.4f} ± {fl_random.std():.4f}")
print(f"Δ = {delta:+.4f}   p = {pval:.2e}")

# ─── 画图 ─────────────────────────────────────────────────────────
COLOR_FIM    = "#2196F3"   # 蓝
COLOR_RANDOM = "#9E9E9E"   # 灰

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                         gridspec_kw=dict(width_ratios=[1.0, 1.3]))

# ──────────────────────────────────────────────────────────────────
# Panel (a): 箱线图 + 散点
# ──────────────────────────────────────────────────────────────────
ax = axes[0]
groups = [fl_top, fl_random]
labels = [f"FIM top-{K_SHOW}\ndirections", "Random\ndirections"]
colors = [COLOR_FIM, COLOR_RANDOM]

bp = ax.boxplot(groups, patch_artist=True, widths=0.45,
                medianprops=dict(color='white', lw=2.5),
                whiskerprops=dict(lw=1.5),
                capprops=dict(lw=1.5),
                flierprops=dict(marker='o', markersize=4, alpha=0.5))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# 叠加抖动散点
rng = np.random.default_rng(42)
for i, (grp, color) in enumerate(zip(groups, colors), start=1):
    jitter = rng.uniform(-0.18, 0.18, size=len(grp))
    ax.scatter(i + jitter, grp, color=color, s=30, alpha=0.8, zorder=5,
               edgecolors='white', linewidths=0.4)

# 均值横线 + 标注
for i, grp in enumerate(groups, start=1):
    ax.hlines(grp.mean(), i - 0.25, i + 0.25,
              colors='#333333', lw=2.0, zorder=6, linestyles='--')

# 显著性括号
y_top = max(fl_top.max(), fl_random.max())
gap   = (fl_top.max() - fl_random.min()) * 0.07
y_br  = y_top + gap * 0.4
h_br  = gap * 0.8
ax.plot([1, 1, 2, 2], [y_br, y_br + h_br, y_br + h_br, y_br],
        lw=1.2, color='black')
stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
ax.text(1.5, y_br + h_br * 1.15, stars + f"\n(p={pval:.1e})",
        ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks([1, 2])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel(r"Low-frequency energy fraction  $f_{\mathrm{low}}$", fontsize=12)
ax.set_title("(a) Distribution of $f_{\\mathrm{low}}$", fontsize=12)
ax.grid(True, axis='y', ls=':', alpha=0.4)
ax_ypad = gap * 0.5
ax.set_ylim(bottom=max(0, fl_random.min() - ax_ypad))

# ──────────────────────────────────────────────────────────────────
# Panel (b): 逐方向折线图（FIM 按特征值排序 vs 随机）
# ──────────────────────────────────────────────────────────────────
ax2 = axes[1]
ranks = np.arange(1, K_SHOW + 1)

# FIM top-K 折线
ax2.plot(ranks, fl_top, color=COLOR_FIM, lw=2.0, marker='o',
         markersize=5, zorder=4,
         label=f"FIM top-{K_SHOW}  (mean={fl_top.mean():.3f})")

# 随机：每个随机方向画一个点（用灰色散点表示）
jitter2 = rng.uniform(-0.2, 0.2, size=K_SHOW)
ax2.scatter(ranks + jitter2, fl_random, color=COLOR_RANDOM,
            s=25, alpha=0.65, zorder=3,
            label=f"Random  (mean={fl_random.mean():.3f})")

# 随机方向的均值 ± std 带
ax2.axhline(fl_random.mean(), color=COLOR_RANDOM, lw=2.0, ls='--', zorder=2)
ax2.fill_between(ranks,
                 fl_random.mean() - fl_random.std(),
                 fl_random.mean() + fl_random.std(),
                 color=COLOR_RANDOM, alpha=0.18, zorder=1,
                 label=f"Random mean ± std")

# Δ 标注（双向箭头）
x_arrow = K_SHOW * 0.55
y_rand  = fl_random.mean()
y_fim   = float(np.interp(x_arrow, ranks, fl_top))
ax2.annotate('', xy=(x_arrow, y_rand + 0.01),
             xytext=(x_arrow, y_fim - 0.01),
             arrowprops=dict(arrowstyle='<->', color='#E53935', lw=1.8))
ax2.text(x_arrow + 0.8, (y_rand + y_fim) / 2,
         f"Δ={delta:+.3f}", color='#E53935',
         fontsize=10, va='center', fontweight='bold')

ax2.set_xlabel(f"FIM eigenvector rank  $j$  (sorted by eigenvalue ↓)",
               fontsize=11)
ax2.set_ylabel(r"$f_{\mathrm{low}}^{(j)}$", fontsize=12)
ax2.set_title("(b) Per-direction $f_{\\mathrm{low}}$ vs. random baseline",
              fontsize=12)
ax2.legend(fontsize=9.5, loc='lower right', framealpha=0.9)
ax2.grid(True, ls=':', alpha=0.4)
ax2.set_xlim(0.3, K_SHOW + 0.7)
ax2.set_xticks(ranks[::4])

# ─── 总标题 ───────────────────────────────────────────────────────
fig.suptitle(
    "FIM principal subspace preferentially captures low-frequency physics\n"
    r"(FNO · $\mathbf{W}_{\mathrm{spectral}}$ layer 2 · Cylinder dataset)",
    fontsize=12, y=1.02)
fig.tight_layout()

for ext in ["pdf", "png"]:
    path = f"{OUTPUT_DIR}/rq1_top20_vs_random.{ext}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
plt.close(fig)
