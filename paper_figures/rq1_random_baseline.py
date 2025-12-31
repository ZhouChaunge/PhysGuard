#!/usr/bin/env python3
"""
FIM vs Random Direction Baseline — Control Experiment for PhysGuard
====================================================================
核心问题：FIM 主子空间真的对低频有偏好，还是只是"流体场本来就低频"的假象？

实验设计（同一层参数，同一模型 FNO Layer 2）：
  ① FIM top-K 方向    (k=1..20, 最重要的 FIM 特征向量)
  ② Random  方向      (K=20 个随机单位向量)
  ③ FIM bottom-K 方向 (最不重要的 FIM 特征向量，需要大量 FIM 样本)

对每个方向 d_i：
  θ' = θ* + ε · d_i → 前向传播 → Δy → FFT → f_low^(i)

预期：f_low(FIM_top) > f_low(Random) >= f_low(FIM_bottom)
      如果三组差异显著 → 证明 FIM 主子空间确实偏向低频

输出：
  figures/rq1_random_baseline.pdf / .png

Run:
  cd RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python \
      -u ../scripts/rq1_random_baseline.py 2>&1 | tee /tmp/rq1_random_baseline.log
"""

import os, sys, logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, mannwhitneyu

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.join(SCRIPT_DIR, "..", "RealPDEBench")
sys.path.insert(0, REPO_DIR)

DATASET_ROOT = "./data/realpdebench/"
CKPT_PATH    = ("./results/001-cylinder"
                "/fno/fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth")
OUTPUT_DIR   = "./figures"

# ── 超参数 ──────────────────────────────────────────────────────────
# FIM 需要足够多的样本才能估计 bottom eigenvectors（需要更多样本）
N_FIM       = 150     # 梯度采样数（更多 → bottom 方向更准确）
N_TEST      = 50      # 测试样本数
K           = 50      # 每组取 50 个方向
N_RANDOM    = 50      # 随机方向个数（与 FIM top/bottom 等量对比）
EPSILON     = 1e-3    # 扰动幅度
FWD_BATCH   = 8

# 使用 FNO 第 2 层（Layer index=1），这是 ρ=-0.93 最强的层
TARGET_PARAM = "spectral_convs.1.weights1"
FNO_KWARGS   = dict(model_name="fno", modes1=4, modes2=12, modes3=16,
                    n_layers=4, width=64)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
#  FFT：2D 空间均值（更高分辨率）
# ═══════════════════════════════════════════════════════════════════

_grid_cache = {}

def _build_grid(H, W, device):
    half = min(H // 2, W // 2)
    jj  = torch.arange(H // 2, device=device).float()
    kk  = torch.arange(W // 2, device=device).float()
    gj, gk = torch.meshgrid(jj, kk, indexing='ij')
    radial = torch.floor(torch.sqrt(gj**2 + gk**2)).long()
    valid  = radial < half
    return radial, valid, half

def measure_f_low(delta_y: torch.Tensor) -> float:
    """
    delta_y: [B, T, H, W, C]
    Returns f_low = 低频能量占比（2D 空间 FFT，时间维度先取均值）
    """
    T, H, W = delta_y.shape[1], delta_y.shape[2], delta_y.shape[3]
    key = (H, W, delta_y.device.type)
    if key not in _grid_cache:
        _grid_cache[key] = _build_grid(H, W, delta_y.device)
    radial, valid, half = _grid_cache[key]

    dy_mean = delta_y.float().mean(dim=1)           # [B, H, W, C]
    D       = torch.fft.fftn(dy_mean, dim=[1, 2])   # [B, H, W, C]
    power   = D.abs() ** 2
    power_h = power[:, :H//2, :W//2, :]             # [B, H//2, W//2, C]

    B, C    = power_h.shape[0], power_h.shape[-1]
    pfv     = power_h.permute(0, 3, 1, 2).reshape(B * C, -1)
    v_flat  = valid.reshape(-1)
    r_flat  = radial.reshape(-1)[v_flat]
    total   = pfv[:, v_flat]

    iLow    = int(round(half / 3))
    low_mask = r_flat < iLow
    e_low   = total[:, low_mask].sum()
    e_total = total.sum()
    return (e_low / (e_total + 1e-30)).item()


# ═══════════════════════════════════════════════════════════════════
#  数据 & 模型
# ═══════════════════════════════════════════════════════════════════

def build_all():
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer
    from realpdebench.model.load_model import load_model

    common   = dict(dataset_name="cylinder", dataset_root=DATASET_ROOT)
    train_ds = CylinderHFDataset(mode="train", dataset_type="numerical", **common)
    val_ds   = CylinderHFDataset(mode="val",   dataset_type="real",      **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)

    model = load_model(train_ds, device=DEVICE, **FNO_KWARGS)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    raw   = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()

    return train_ds, val_ds, normalizer, model


# ═══════════════════════════════════════════════════════════════════
#  FIM 特征向量（top 和 bottom）
# ═══════════════════════════════════════════════════════════════════

def collect_grads(model, loader, normalizer, n):
    grads = []
    pbar  = tqdm(total=n, desc="  FIM grads", leave=False)
    cnt   = 0
    for x, y in loader:
        if cnt >= n: break
        x, _ = normalizer.preprocess(x, y)
        x    = x[:1].to(DEVICE)
        model.zero_grad()
        model(x).pow(2).mean().backward()
        for pname, p in model.named_parameters():
            if pname != TARGET_PARAM: continue
            if p.grad is None: break
            if p.is_complex():
                gv = torch.cat([p.grad.real.reshape(-1),
                                p.grad.imag.reshape(-1)]).float().cpu()
            else:
                gv = p.grad.reshape(-1).float().cpu()
            grads.append(gv)
            break
        cnt += 1; pbar.update(1)
    pbar.close()
    log.info(f"  Collected {len(grads)} gradient vectors, dim={grads[0].shape[0]}")
    return grads


def compute_eigenvectors(grads, k_top, k_bottom):
    """
    返回 (U_top, U_bottom, eigenvalues_all)
    U_top:    [dim, k_top]   — 最重要的 k_top 个方向
    U_bottom: [dim, k_bottom] — 最不重要的 k_bottom 个方向
    """
    J   = torch.stack(grads, dim=0)          # [N, dim]
    G   = J @ J.T                            # [N, N] Gram matrix
    L, V = np.linalg.eigh(G.numpy().astype(np.float64))
    L   = np.maximum(L[::-1], 0); V = V[:, ::-1]

    n_valid = int((L > 1e-12 * L[0]).sum())
    log.info(f"  Valid eigenvectors: {n_valid} (out of {len(L)})")
    log.info(f"  λ_1={L[0]:.3e}  λ_{n_valid}={L[n_valid-1]:.3e}")

    def make_u(j):
        lam = L[j]
        if lam < 1e-30: return None
        vj = torch.tensor(V[:, j], dtype=torch.float32)
        uj = J.T @ vj / (lam ** 0.5)
        uj = uj / (uj.norm() + 1e-12)
        return uj

    # ── top-k ──
    U_top_list = []
    for j in range(k_top):
        u = make_u(j)
        if u is not None: U_top_list.append(u)
    U_top = torch.stack(U_top_list, dim=1)

    # ── bottom-k: 取最末尾 k_bottom 个有效特征向量 ──
    # 注意：已经 flip 过了，所以末尾是最小特征值
    bottom_start = max(0, n_valid - k_bottom)
    U_bot_list = []
    for j in range(bottom_start, n_valid):
        u = make_u(j)
        if u is not None: U_bot_list.append(u)
    # 如果有效数量不足 k_bottom，补随机
    while len(U_bot_list) < k_bottom:
        v = torch.randn(J.shape[1])
        v = v / (v.norm() + 1e-12)
        U_bot_list.append(v)
    U_bot = torch.stack(U_bot_list[-k_bottom:], dim=1)

    log.info(f"  U_top: {tuple(U_top.shape)}  U_bot: {tuple(U_bot.shape)}")
    return U_top, U_bot, L[:n_valid]


def make_random_directions(dim, k):
    """生成 k 个随机单位向量（在同一参数维度）"""
    R = torch.randn(dim, k)
    R = R / (R.norm(dim=0, keepdim=True) + 1e-12)
    return R


# ═══════════════════════════════════════════════════════════════════
#  基础输出 & 扰动测量
# ═══════════════════════════════════════════════════════════════════

def get_base_outputs(model, loader, normalizer, n):
    xs, ys = [], []
    cnt = 0
    with torch.no_grad():
        for x, y in loader:
            if cnt >= n: break
            x, _ = normalizer.preprocess(x, y)
            rem  = min(n - cnt, x.shape[0])
            x    = x[:rem]
            xs.append(x.cpu()); ys.append(model(x.to(DEVICE)).cpu())
            cnt += rem
    return torch.cat(xs), torch.cat(ys)


def sweep_directions(model, directions, x_test, y_base, label):
    """
    directions: [dim, K] 单位向量矩阵
    返回 f_low 列表，长度 K
    """
    K = directions.shape[1]
    results = []
    for j in tqdm(range(K), desc=f"  sweep {label}", leave=True):
        u_j = directions[:, j]
        for pname, p in model.named_parameters():
            if pname != TARGET_PARAM: continue
            if p.is_complex():
                d = p.numel()
                re = u_j[:d].reshape(p.shape).to(p.device)
                im = u_j[d:].reshape(p.shape).to(p.device)
                delta = torch.complex(re, im).to(dtype=p.dtype)
            else:
                delta = u_j.reshape(p.shape).to(device=p.device, dtype=p.dtype)

            with torch.no_grad(): p.data.add_(EPSILON * delta)

            ys = []
            with torch.no_grad():
                for st in range(0, x_test.shape[0], FWD_BATCH):
                    ys.append(model(x_test[st:st+FWD_BATCH].to(DEVICE)).cpu())
            y_pert = torch.cat(ys)

            with torch.no_grad(): p.data.sub_(EPSILON * delta)

            delta_y = y_pert - y_base
            fl = measure_f_low(delta_y)
            results.append(fl)
            break

    results = np.array(results)
    log.info(f"  [{label}] mean={results.mean():.4f}  std={results.std():.4f}  "
             f"min={results.min():.4f}  max={results.max():.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════
#  统计检验
# ═══════════════════════════════════════════════════════════════════

def stats_test(a, b, name_a, name_b):
    stat, pval = mannwhitneyu(a, b, alternative='greater')
    log.info(f"  Mann-Whitney U ({name_a} > {name_b}): U={stat:.1f}  p={pval:.4e}  "
             + ("✓ significant" if pval < 0.05 else "✗ not significant"))
    return pval


# ═══════════════════════════════════════════════════════════════════
#  图
# ═══════════════════════════════════════════════════════════════════

def make_figure(fl_top, fl_random, fl_bottom, pval_top_vs_random, pval_random_vs_bottom):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw=dict(width_ratios=[1.2, 1.0]))

    # ── Panel (a): 分组箱线图 + 散点 ────────────────────────────────
    ax = axes[0]
    groups  = [fl_top, fl_random, fl_bottom]
    labels  = ["FIM top-20\n(most important)",
               "Random\n(baseline)",
               "FIM bottom-20\n(least important)"]
    colors  = ["#2196F3", "#9E9E9E", "#FF5722"]

    bp = ax.boxplot(groups, patch_artist=True, widths=0.45,
                    medianprops=dict(color='white', lw=2.5),
                    whiskerprops=dict(lw=1.5),
                    capprops=dict(lw=1.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # 叠加散点（jitter）
    rng = np.random.default_rng(0)
    for i, (grp, color) in enumerate(zip(groups, colors), start=1):
        jitter = rng.uniform(-0.18, 0.18, size=len(grp))
        ax.scatter(i + jitter, grp, color=color, s=20, alpha=0.7, zorder=5)

    # 显著性括号
    def bracket(ax, x1, x2, y, h, pval):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color='black')
        stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else
                ("*" if pval < 0.05 else "n.s."))
        ax.text((x1+x2)/2, y+h*1.1, stars,
                ha='center', va='bottom', fontsize=12)

    y_top  = max(fl_top.max(), fl_random.max(), fl_bottom.max())
    gap    = (max(fl_top.max(), fl_random.max()) - min(fl_random.min(), fl_bottom.min())) * 0.06
    bracket(ax, 1, 2, y_top + gap*0.5, gap, pval_top_vs_random)
    bracket(ax, 2, 3, y_top + gap*0.5, gap, pval_random_vs_bottom)
    bracket(ax, 1, 3, y_top + gap*2.5, gap, stats_test(fl_top, fl_bottom,
                                                         "FIM-top", "FIM-bottom"))

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"Low-freq energy fraction $f_{\mathrm{low}}$", fontsize=12)
    ax.set_title("(a) $f_{\\mathrm{low}}$ by direction type\n"
                 "(FNO · spectral_convs.1 · Cylinder)", fontsize=11)
    ax.grid(True, axis='y', ls=':', alpha=0.4)
    ax.set_ylim(bottom=max(0, min(fl_bottom.min(), fl_random.min()) - gap))

    # ── Panel (b): 按排名 j 的逐点折线图（top vs random vs bottom 各方向） ──
    ax2 = axes[1]
    ranks = np.arange(1, K + 1)

    ax2.plot(ranks, fl_top,    color=colors[0], lw=2,   marker='o',
             markersize=5, label=f"FIM top-20 (mean={fl_top.mean():.3f})")
    ax2.axhline(fl_random.mean(), color=colors[1], lw=2, ls='--',
                label=f"Random mean={fl_random.mean():.3f}")
    ax2.fill_between(ranks,
                     fl_random.mean() - fl_random.std(),
                     fl_random.mean() + fl_random.std(),
                     color=colors[1], alpha=0.15)
    ax2.plot(ranks, fl_bottom, color=colors[2], lw=2,   marker='s',
             markersize=5, label=f"FIM bottom-20 (mean={fl_bottom.mean():.3f})")

    ax2.set_xlabel("Direction index $j$ (FIM: ranked by eigenvalue)", fontsize=11)
    ax2.set_ylabel(r"$f_{\mathrm{low}}^{(j)}$", fontsize=12)
    ax2.set_title("(b) Per-direction $f_{\\mathrm{low}}$\n"
                  "with random baseline band", fontsize=11)
    ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax2.grid(True, ls=':', alpha=0.4)
    ax2.set_xlim(0.5, K + 0.5)
    ax2.set_xticks([1, 10, 20, 30, 40, 50])

    fig.suptitle(
        "FIM principal subspace preferentially captures low-frequency physics\n"
        r"($f_{\mathrm{low}}$: FIM top-20 vs. random directions vs. FIM bottom-20)",
        fontsize=12, y=1.01)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = f"{OUTPUT_DIR}/rq1_random_baseline.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("RQ1 Random Baseline Control Experiment")
    log.info(f"  Layer: {TARGET_PARAM}")
    log.info(f"  N_FIM={N_FIM}  N_TEST={N_TEST}  K={K}  ε={EPSILON}")
    log.info("=" * 60)

    log.info("Loading data & model …")
    train_ds, val_ds, normalizer, model = build_all()
    log.info(f"  train={len(train_ds)}  val={len(val_ds)}")

    loader_fim  = DataLoader(train_ds, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    loader_test = DataLoader(val_ds,   batch_size=FWD_BATCH, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── 1. FIM 梯度 ──────────────────────────────────────────────
    log.info("\nStep 1: Computing FIM gradients …")
    grads = collect_grads(model, loader_fim, normalizer, N_FIM)

    # ── 2. 特征向量 (top & bottom) ──────────────────────────────
    log.info("\nStep 2: Computing eigenvectors (top + bottom) …")
    U_top, U_bottom, eigenvalues = compute_eigenvectors(grads, K, K)
    dim = U_top.shape[0]

    # ── 3. 随机方向 ──────────────────────────────────────────────
    log.info("\nStep 3: Generating random unit directions …")
    U_random = make_random_directions(dim, N_RANDOM)
    log.info(f"  Random directions: {tuple(U_random.shape)}")

    # ── 4. 基础输出 ──────────────────────────────────────────────
    log.info("\nStep 4: Computing base model outputs …")
    x_test, y_base = get_base_outputs(model, loader_test, normalizer, N_TEST)
    log.info(f"  x_test: {tuple(x_test.shape)}")

    # ── 5. 扰动 & 测量 ───────────────────────────────────────────
    log.info("\nStep 5: Sweeping FIM top-20 directions …")
    fl_top = sweep_directions(model, U_top, x_test, y_base, "FIM-top")

    log.info("\nStep 6: Sweeping random directions …")
    fl_random = sweep_directions(model, U_random, x_test, y_base, "Random")

    log.info("\nStep 7: Sweeping FIM bottom-20 directions …")
    fl_bottom = sweep_directions(model, U_bottom, x_test, y_base, "FIM-bottom")

    # ── 6. 统计检验 ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"  FIM top-20   f_low: {fl_top.mean():.4f} ± {fl_top.std():.4f}")
    log.info(f"  Random       f_low: {fl_random.mean():.4f} ± {fl_random.std():.4f}")
    log.info(f"  FIM bottom-20 f_low: {fl_bottom.mean():.4f} ± {fl_bottom.std():.4f}")

    log.info("\nStatistical tests (one-sided Mann-Whitney U):")
    p1 = stats_test(fl_top,    fl_random, "FIM-top",    "Random")
    p2 = stats_test(fl_random, fl_bottom, "Random",     "FIM-bottom")
    p3 = stats_test(fl_top,    fl_bottom, "FIM-top",    "FIM-bottom")

    diff_top_rand  = fl_top.mean() - fl_random.mean()
    diff_rand_bot  = fl_random.mean() - fl_bottom.mean()
    log.info(f"\n  Δ(FIM-top vs Random)     = {diff_top_rand:+.4f}")
    log.info(f"  Δ(Random vs FIM-bottom)  = {diff_rand_bot:+.4f}")
    log.info(f"  Δ(FIM-top vs FIM-bottom) = {fl_top.mean()-fl_bottom.mean():+.4f}")

    # ── 7. 保存数据 ──────────────────────────────────────────────
    np.savez(f"{OUTPUT_DIR}/rq1_random_baseline_data.npz",
             fl_top=fl_top, fl_random=fl_random, fl_bottom=fl_bottom,
             p_top_vs_random=p1, p_random_vs_bottom=p2, p_top_vs_bottom=p3)

    # ── 8. 画图 ──────────────────────────────────────────────────
    log.info("\nPlotting …")
    make_figure(fl_top, fl_random, fl_bottom, p1, p2)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
