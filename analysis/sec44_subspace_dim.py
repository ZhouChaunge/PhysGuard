"""
§4.4 Fisher Subspace Dimensionality Statistics
================================================
对所有架构读取 null_space_projections.pt，统计：
    - 每层的参数维度 d_m
    - 自适应选出的子空间维度 k
    - 比例 k/d_m（极低 → 物理知识高度集中）

输出：
    analysis/figures/fig_subspace_dim.pdf / .png
    控制台打印 LaTeX 格式的表格
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROJ_FILES = {
    "FNO":       "./results/001-cylinder/fno/fno_cylinder_nsft/2026-03-10_08-42-11/null_space_projections.pt",
    "CNO":       "./results/001-cylinder/cno/cno_cylinder_nsft/2026-03-14_06-37-42/null_space_projections.pt",
    "DeepONet":  "./results/001-cylinder/deeponet/deeponet_cylinder_nsft/2026-03-13_00-38-45/null_space_projections.pt",
    "DPOT":      "./results/001-cylinder/dpot/dpot_s_cylinder_nsft/2026-03-14_06-37-42/null_space_projections.pt",
    "Transolver": "./results/001-cylinder/transolver/transolver_cylinder_nsft/2026-03-30_14-04-24/null_space_projections.pt",
}

# 每个架构中最具代表性的层（参数量最大的层）
REPRESENTATIVE_LAYER = {
    "FNO":       "spectral_convs.0.weights1",
    "CNO":       None,  # 自动选最大
    "DeepONet":  None,
    "DPOT":      None,
    "Transolver": None,
}


def load_proj(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["projections"], data.get("n_components", "?"), data.get("alpha", "?")


def summarize(projections):
    """返回 (总参数量, 总k, 各层统计列表)"""
    rows = []
    for name, pdata in projections.items():
        U = pdata["U"]
        d = pdata.get("dim", U.shape[0])
        k = U.shape[1]
        rows.append({"name": name, "d": d, "k": k, "ratio": k / d})
    rows.sort(key=lambda x: -x["d"])
    total_d = sum(r["d"] for r in rows)
    total_k = sum(r["k"] for r in rows)
    return total_d, total_k, rows


def main():
    all_summary = {}
    for arch, path in PROJ_FILES.items():
        if not Path(path).exists():
            print(f"  [{arch}] SKIP — file not found")
            continue
        proj, n_comp, alpha = load_proj(path)
        total_d, total_k, rows = summarize(proj)
        all_summary[arch] = {
            "total_d": total_d, "total_k": total_k,
            "rows": rows, "n_comp": n_comp, "alpha": alpha
        }
        print(f"[{arch}] n_layers={len(rows)}, total_params={total_d:,}, "
              f"total_k={total_k}, avg_ratio={total_k/total_d*100:.4f}%")

    # ── 图1：各架构最大层的 k/d_m 柱状图 ──────────────────────────────────────
    # 选每个架构中 d 最大的层
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    archs = list(all_summary.keys())

    # 左图：最大单层的 k 与 d_m
    ax = axes[0]
    rep_data = []
    for arch in archs:
        rows = all_summary[arch]["rows"]
        rep = rows[0]  # 已按 d 降序排列，取最大层
        rep_data.append((arch, rep["name"].split(".")[-1], rep["d"], rep["k"]))

    x = np.arange(len(rep_data))
    w = 0.38
    d_vals = [r[2] for r in rep_data]
    k_vals = [r[3] for r in rep_data]

    bars_d = ax.bar(x - w/2, d_vals, w, label="$d_m$ (param dim)", color="#90CAF9", log=True)
    bars_k = ax.bar(x + w/2, k_vals, w, label="$k$ (subspace dim)", color="#1E88E5", log=True)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rep_data], fontsize=11)
    ax.set_ylabel("Dimension (log scale)", fontsize=10)
    ax.set_title("Largest layer: $d_m$ vs. selected $k$\n(log scale)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 在 k bar 上标注 k/d 比例
    for bar, (_, _, d, k) in zip(bars_k, rep_data):
        ratio_pct = k / d * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                f"{ratio_pct:.4f}%", ha="center", va="bottom", fontsize=8,
                color="#1E88E5", fontweight="bold")

    # 右图：各架构所有层的 k/d 比例分布（violin / scatter）
    ax2 = axes[1]
    colors = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]
    for i, arch in enumerate(archs):
        rows = all_summary[arch]["rows"]
        ratios = [r["ratio"] * 100 for r in rows]  # as percent
        x_jitter = np.random.normal(i, 0.08, len(ratios))
        ax2.scatter(x_jitter, ratios, alpha=0.6, s=25,
                    color=colors[i % len(colors)], label=arch)
        ax2.scatter([i], [np.median(ratios)], s=100, color=colors[i % len(colors)],
                    marker="D", zorder=5, edgecolors="black", linewidths=0.8)

    ax2.set_xticks(range(len(archs)))
    ax2.set_xticklabels(archs, fontsize=11)
    ax2.set_yscale("log")
    ax2.set_ylabel("k / d  (%)  — log scale", fontsize=10)
    ax2.set_title("Subspace ratio $k/d_m$ across all protected layers\n"
                  "(diamond = median; lower = more concentrated)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    plt.suptitle(
        "Fisher subspace is extremely low-dimensional\n"
        "(largest layer $k/d_m$ ranges from 0.0001% to 0.004% across architectures)",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    out_pdf = OUT_DIR / "fig_subspace_dim.pdf"
    out_png = OUT_DIR / "fig_subspace_dim.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\n图片保存：{out_png}")

    # ── 打印 LaTeX 表格 ────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("LaTeX 表格（paper §4.4 用）：")
    print("="*75)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Adaptive subspace dimensionality ($\tau=0.9$, $N=200$).}")
    print(r"\small")
    print(r"\begin{tabular}{llrrl}")
    print(r"\toprule")
    print(r"Architecture & Representative layer & $d_m$ & $k$ & $k/d_m$ \\")
    print(r"\midrule")
    for arch in archs:
        rows   = all_summary[arch]["rows"]
        rep    = rows[0]
        layer  = rep["name"]
        # 简化层名
        layer_short = ".".join(layer.split(".")[-2:]) if "." in layer else layer
        d = rep["d"]
        k = rep["k"]
        ratio  = k / d
        ratio_str = f"{ratio*100:.4f}\\%"
        d_str = f"{d:,}".replace(",", "{,}")
        print(f"{arch} & \\texttt{{{layer_short}}} & ${d_str}$ & ${k}$ & {ratio_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*75)

    # ── 关键数字摘要 ──────────────────────────────────────────────────────────
    print("\n关键数字：")
    for arch in archs:
        rows = all_summary[arch]["rows"]
        rep  = rows[0]
        print(f"  {arch}: 最大层 k={rep['k']} / d_m={rep['d']:,} = {rep['ratio']*100:.6f}%")


if __name__ == "__main__":
    main()
