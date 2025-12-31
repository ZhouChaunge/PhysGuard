#!/usr/bin/env python3
"""
Multi-Architecture FIM vs Random Direction Baseline
======================================================
在 FNO / CNO / DeepONet / DPOT 四种架构上分别验证：
FIM 主子空间 top-K 方向的输出扰动显著偏向低频物理模态，
而同维度随机方向的扰动则充满高频噪声。

FNO 的数据直接从已有的 rq1_random_baseline_data.npz 中复用，
其余三种架构在本脚本中现场计算。

Run:
  cd ./RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python \
      -u ../scripts/rq1_multi_arch_vs_random.py 2>&1 | tee /tmp/rq1_multi_arch.log
"""

import os, sys, logging, pickle
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu

# ─── 路径设置 ─────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.join(SCRIPT_DIR, "..", "RealPDEBench")
sys.path.insert(0, REPO_DIR)

DATASET_ROOT = "./data/realpdebench/"
OUTPUT_DIR   = "./figures"
CKPT_BASE    = "./results/001-cylinder"
CACHE_DIR    = "/tmp/rq1_multi_arch_cache"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── 超参数 ───────────────────────────────────────────────────────
K        = 20    # FIM top-K 方向 / 随机方向数量
N_FIM    = 80    # FIM 梯度采样数（每个架构，减小以加速）
N_TEST   = 30    # 测试样本数
EPSILON  = 1e-3
FWD_BATCH = 4
SEED     = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── 各架构配置 ───────────────────────────────────────────────────
ARCH_CONFIGS = {
    "FNO": {
        "ckpt": f"{CKPT_BASE}/fno/fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth",
        "build_kwargs": dict(model_name="fno", modes1=4, modes2=12, modes3=16,
                             n_layers=4, width=64),
        "target_param": "spectral_convs.1.weights1",  # 已验证最有效的层
        "cached_npz": f"{OUTPUT_DIR}/rq1_random_baseline_data.npz",  # 复用已有结果
    },
    "CNO": {
        "ckpt": f"{CKPT_BASE}/cno/cno_cylinder_pretrained/2026-03-11_09-49-04/model_5000.pth",
        "build_kwargs": dict(model_name="cno", N_layers=3),
        "target_param": None,   # 自动选最大层
        "cached_npz": None,
    },
    "DeepONet": {
        "ckpt": (f"{CKPT_BASE}/deeponet/deeponet_cylinder_pretrained"
                 "/2026-03-12_11-16-03/model_5000.pth"),
        "build_kwargs": dict(model_name="deeponet", p=128, dropout_rate=0.1),
        # trunk.fc.4 是 trunk MLP 的最终层 Linear(128→128)，决定空间基函数的输出
        # 相比 branch.fc.0（全局池化后），这层的扰动直接影响空间坐标→基函数的映射
        "target_param": "trunk.fc.4.weight",
        "cached_npz": None,
    },
    "Transolver": {
        "ckpt": (f"{CKPT_BASE}/transolver/transolver_cylinder_pretrained"
                 "/2026-03-25_21-23-52/model_5000.pth"),
        "build_kwargs": dict(
            model_name="transolver",
            space_dim=3, n_layers=1, n_hidden=256, n_head=8,
            H=128, W=64, D=20,
            fun_dim=0, out_dim=3, ref=4,
            dropout=0.1, act="gelu", mlp_ratio=4, slice_num=16,
        ),
        "target_param": None,
        "cached_npz": None,
        "fwd_batch": 1,  # Transolver 大中间张量，batch>1 OOM
    },
}

# 视觉样式
ARCH_COLOR  = {"FNO": "#4C72B0", "CNO": "#DD8452",
               "DeepONet": "#55A868", "Transolver": "#8172B2"}
ARCH_MARKER = {"FNO": "o", "CNO": "s", "DeepONet": "^", "Transolver": "P"}


# ═══════════════════════════════════════════════════════════════════
#  FFT 工具
# ═══════════════════════════════════════════════════════════════════

_grid_cache = {}

def _build_grid(H, W, device):
    half  = min(H // 2, W // 2)
    jj    = torch.arange(H // 2, device=device).float()
    kk    = torch.arange(W // 2, device=device).float()
    gj, gk = torch.meshgrid(jj, kk, indexing='ij')
    radial = torch.floor(torch.sqrt(gj**2 + gk**2)).long()
    valid  = radial < half
    return radial, valid, half


def measure_f_low(delta_y: torch.Tensor) -> float:
    """delta_y: [B, T, H, W, C] → 低频能量占比（2D 空间 FFT）"""
    H, W = delta_y.shape[2], delta_y.shape[3]
    key  = (H, W, str(delta_y.device))
    if key not in _grid_cache:
        _grid_cache[key] = _build_grid(H, W, delta_y.device)
    radial, valid, half = _grid_cache[key]

    dy_mean = delta_y.float().mean(dim=1)            # [B, H, W, C]
    D       = torch.fft.fftn(dy_mean, dim=[1, 2])    # [B, H, W, C]
    power   = D.abs() ** 2
    power_h = power[:, :H//2, :W//2, :]

    B, C  = power_h.shape[0], power_h.shape[-1]
    pfv   = power_h.permute(0, 3, 1, 2).reshape(B * C, -1)
    v_flat = valid.reshape(-1)
    r_flat = radial.reshape(-1)[v_flat]
    total  = pfv[:, v_flat]

    iLow    = int(round(half / 3))
    low_mask = r_flat < iLow
    e_low   = total[:, low_mask].sum()
    e_total = total.sum()
    return (e_low / (e_total + 1e-30)).item()


# ═══════════════════════════════════════════════════════════════════
#  模型工具
# ═══════════════════════════════════════════════════════════════════

def real_numel(p):
    return 2 * p.numel() if p.is_complex() else p.numel()


def select_largest_param(model):
    """返回最大 weight 矩阵的名称（排除 bias / norm / embed 等）。"""
    candidates = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.dim() < 2:
            continue
        low = name.lower()
        if any(tok in low for tok in ("bias", "norm", ".bn", "embed", "placeholder")):
            continue
        candidates.append((name, real_numel(p)))
    candidates.sort(key=lambda x: -x[1])
    chosen = candidates[0][0] if candidates else None
    log.info(f"  Auto-selected target param: {chosen} "
             f"({candidates[0][1] if candidates else 0} real scalars)")
    return chosen


def build_and_load(arch_name, train_ds):
    from realpdebench.model.load_model import load_model
    cfg   = ARCH_CONFIGS[arch_name]["build_kwargs"]
    model = load_model(train_ds, device=DEVICE, **cfg)
    ckpt  = torch.load(ARCH_CONFIGS[arch_name]["ckpt"], map_location=DEVICE)
    raw   = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


# ═══════════════════════════════════════════════════════════════════
#  FIM & 随机方向
# ═══════════════════════════════════════════════════════════════════

def collect_grads(model, loader, normalizer, target_name, n):
    grads = []
    pbar  = tqdm(total=n, desc="  FIM grads", leave=False)
    cnt   = 0
    for x, y in loader:
        if cnt >= n:
            break
        x, _ = normalizer.preprocess(x, y)
        x     = x[:1].to(DEVICE)
        model.zero_grad()
        loss  = model(x).pow(2).mean()
        loss.backward()
        for pname, p in model.named_parameters():
            if pname != target_name:
                continue
            if p.grad is None:
                break
            if p.is_complex():
                gv = torch.cat([p.grad.real.reshape(-1),
                                p.grad.imag.reshape(-1)]).float().cpu()
            else:
                gv = p.grad.reshape(-1).float().cpu()
            grads.append(gv)
            break
        cnt += 1
        pbar.update(1)
    pbar.close()
    return grads


def compute_top_eigenvectors(grads, k):
    J       = torch.stack(grads, dim=0)   # [N, dim]
    G       = J @ J.T
    L, V    = np.linalg.eigh(G.numpy().astype(np.float64))
    L       = np.maximum(L[::-1], 0); V = V[:, ::-1]
    n_valid = int((L > 1e-12 * L[0]).sum())
    log.info(f"  Valid eigenvectors: {n_valid} / {len(L)}")
    log.info(f"  λ_1={L[0]:.3e}  λ_{n_valid}={L[n_valid-1]:.3e}")

    U_list = []
    for j in range(min(k, n_valid)):
        lam = L[j]
        if lam < 1e-30:
            continue
        vj = torch.tensor(V[:, j], dtype=torch.float32)
        uj = J.T @ vj / (lam ** 0.5)
        uj = uj / (uj.norm() + 1e-12)
        U_list.append(uj)

    if len(U_list) < k:
        log.warning(f"  Only {len(U_list)} valid eigenvectors (wanted {k}), padding random")
        dim = grads[0].shape[0]
        while len(U_list) < k:
            v = torch.randn(dim)
            U_list.append(v / (v.norm() + 1e-12))

    U = torch.stack(U_list[:k], dim=1)
    log.info(f"  U_top shape: {tuple(U.shape)}")
    return U


def make_random_dirs(dim, k):
    R = torch.randn(dim, k)
    return R / (R.norm(dim=0, keepdim=True) + 1e-12)


# ═══════════════════════════════════════════════════════════════════
#  扰动测量
# ═══════════════════════════════════════════════════════════════════

def get_base_outputs(model, loader, normalizer, n):
    xs, ys = [], []
    cnt    = 0
    with torch.no_grad():
        for x, y in loader:
            if cnt >= n:
                break
            x, _ = normalizer.preprocess(x, y)
            rem   = min(n - cnt, x.shape[0])
            xs.append(x[:rem].cpu())
            ys.append(model(x[:rem].to(DEVICE)).cpu())
            cnt  += rem
    return torch.cat(xs), torch.cat(ys)


def sweep(model, target_name, directions, x_test, y_base, label, fwd_batch=FWD_BATCH):
    K_  = directions.shape[1]
    results = []
    for j in tqdm(range(K_), desc=f"  {label}", leave=True):
        u_j = directions[:, j]
        for pname, p in model.named_parameters():
            if pname != target_name:
                continue
            if p.is_complex():
                d  = p.numel()
                re = u_j[:d].reshape(p.shape).to(p.device)
                im = u_j[d:].reshape(p.shape).to(p.device)
                delta = torch.complex(re, im).to(dtype=p.dtype)
            else:
                delta = u_j.reshape(p.shape).to(device=p.device, dtype=p.dtype)

            with torch.no_grad():
                p.data.add_(EPSILON * delta)

            ys = []
            with torch.no_grad():
                for st in range(0, x_test.shape[0], fwd_batch):
                    ys.append(model(x_test[st:st+fwd_batch].to(DEVICE)).cpu())
            y_pert = torch.cat(ys)

            with torch.no_grad():
                p.data.sub_(EPSILON * delta)

            fl = measure_f_low(y_pert - y_base)
            results.append(fl)
            break

    results = np.array(results)
    log.info(f"  [{label}] mean={results.mean():.4f}  std={results.std():.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════
#  每个架构的完整流程
# ═══════════════════════════════════════════════════════════════════

def run_arch(arch_name, train_ds, val_ds, normalizer):
    cache_path = os.path.join(CACHE_DIR, f"{arch_name}.npz")

    # FNO 直接复用已有结果
    cfg = ARCH_CONFIGS[arch_name]
    if cfg["cached_npz"] is not None and os.path.exists(cfg["cached_npz"]):
        log.info(f"  [{arch_name}] Loading cached data from {cfg['cached_npz']}")
        d = np.load(cfg["cached_npz"])
        fl_top    = d["fl_top"][:K]
        fl_random = d["fl_random"][:K]
        return fl_top, fl_random

    # 先检查本地 cache
    if os.path.exists(cache_path):
        log.info(f"  [{arch_name}] Loading local cache: {cache_path}")
        d = np.load(cache_path)
        return d["fl_top"], d["fl_random"]

    log.info(f"\n{'='*50}")
    log.info(f"  [{arch_name}] Building model …")
    model = build_and_load(arch_name, train_ds)

    target = cfg["target_param"]
    if target is None:
        target = select_largest_param(model)
    else:
        log.info(f"  [{arch_name}] Fixed target param: {target}")

    # 验证 target 可访问
    found = any(n == target for n, _ in model.named_parameters())
    if not found:
        log.warning(f"  [{arch_name}] target '{target}' not found! auto-selecting.")
        target = select_largest_param(model)

    # 读取每个架构专属的 fwd_batch（Transolver 需要 batch=1 避免 OOM）
    arch_fwd_batch = cfg.get("fwd_batch", FWD_BATCH)
    log.info(f"  [{arch_name}] fwd_batch={arch_fwd_batch}")

    loader_fim  = DataLoader(train_ds, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    loader_test = DataLoader(val_ds, batch_size=arch_fwd_batch, shuffle=False,
                             num_workers=4, pin_memory=True)

    log.info(f"  [{arch_name}] Collecting FIM gradients (n={N_FIM}) …")
    grads = collect_grads(model, loader_fim, normalizer, target, N_FIM)

    log.info(f"  [{arch_name}] Computing FIM eigenvectors (top-{K}) …")
    U_top    = compute_top_eigenvectors(grads, K)
    dim      = U_top.shape[0]

    log.info(f"  [{arch_name}] Generating {K} random directions …")
    U_random = make_random_dirs(dim, K)

    log.info(f"  [{arch_name}] Getting base outputs (n={N_TEST}) …")
    x_test, y_base = get_base_outputs(model, loader_test, normalizer, N_TEST)

    log.info(f"  [{arch_name}] Sweeping FIM top-{K} …")
    fl_top    = sweep(model, target, U_top,    x_test, y_base,
                     f"{arch_name} FIM-top", fwd_batch=arch_fwd_batch)

    log.info(f"  [{arch_name}] Sweeping Random …")
    fl_random = sweep(model, target, U_random, x_test, y_base,
                     f"{arch_name} Random", fwd_batch=arch_fwd_batch)

    np.savez(cache_path, fl_top=fl_top, fl_random=fl_random)
    log.info(f"  [{arch_name}] Cached to {cache_path}")

    # 释放 GPU 内存
    del model
    torch.cuda.empty_cache()

    return fl_top, fl_random


# ═══════════════════════════════════════════════════════════════════
#  绘图（重新设计，更美观）
# ═══════════════════════════════════════════════════════════════════

def make_figure(results: dict):
    """
    results: {arch_name: (fl_top [K], fl_random [K])}
    双面板图：
      (a) Slope chart（"哑铃"图）：每个架构画两点+连线，Random→FIM
      (b) Delta 条形图（每个架构的均值差）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.family"    : "DejaVu Sans",
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.grid"     : True,
        "grid.alpha"    : 0.25,
        "grid.linestyle": ":",
    })

    archs = list(results.keys())
    n     = len(archs)

    fig = plt.figure(figsize=(12, 4.8))
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[2.0, 0.08, 1.0],
                   wspace=0.04)
    ax_main = fig.add_subplot(gs[0])
    ax_gap  = fig.add_subplot(gs[2])

    # ── Panel (a): Slope / dumbbell chart ─────────────────────────
    x_rand = 0.0
    x_fim  = 1.0

    for i, arch in enumerate(archs):
        fl_top, fl_rand = results[arch]
        color  = ARCH_COLOR[arch]
        marker = ARCH_MARKER[arch]

        mu_fim  = fl_top.mean()
        mu_rand = fl_rand.mean()
        se_fim  = fl_top.std() / np.sqrt(len(fl_top))
        se_rand = fl_rand.std() / np.sqrt(len(fl_rand))

        # 连接线（略带透明）
        ax_main.plot([x_rand, x_fim], [mu_rand, mu_fim],
                     color=color, lw=2.0, alpha=0.7, zorder=2)

        # 误差棒
        ax_main.errorbar(x_rand, mu_rand,
                         yerr=2 * se_rand,        # ≈ 95% CI
                         fmt=marker, color=color,
                         markersize=10, capsize=4, lw=1.5, zorder=4,
                         markerfacecolor='white', markeredgewidth=2.0)
        ax_main.errorbar(x_fim, mu_fim,
                         yerr=2 * se_fim,
                         fmt=marker, color=color,
                         markersize=10, capsize=4, lw=1.5, zorder=4,
                         label=arch)

        # 在 FIM 侧加文字标注
        ax_main.text(x_fim + 0.025, mu_fim,
                     f"{mu_fim:.3f}", va='center',
                     fontsize=9, color=color, fontweight='bold')

    ax_main.set_xticks([x_rand, x_fim])
    ax_main.set_xticklabels(["Random\ndirections",
                              f"FIM top-{K}\ndirections"],
                             fontsize=12)
    ax_main.set_xlim(-0.15, 1.25)
    ax_main.set_ylabel(r"Low-frequency energy fraction  $f_{\mathrm{low}}$",
                       fontsize=12)
    ax_main.set_title(r"(a) $f_{\mathrm{low}}$: FIM vs. random directions"
                      "\n(each line = one architecture)",
                      fontsize=11)
    ax_main.legend(loc='center left', fontsize=9.5, framealpha=0.9,
                   handlelength=1.0)
    ax_main.set_ylim(-0.02, 1.08)

    # 添加水平参考线
    ax_main.axhline(1.0, color='#cccccc', lw=1.0, ls='--', zorder=1)
    ax_main.axhline(0.0, color='#cccccc', lw=1.0, ls='--', zorder=1)

    # ── Panel (b): Δ 条形图 ────────────────────────────────────────
    y_pos = np.arange(n)[::-1]   # 反序让 FNO 在顶部
    deltas, pvals, colors = [], [], []

    for arch in archs:
        fl_top, fl_rand = results[arch]
        delta  = fl_top.mean() - fl_rand.mean()
        _, pval = mannwhitneyu(fl_top, fl_rand, alternative='greater')
        deltas.append(delta)
        pvals.append(pval)
        colors.append(ARCH_COLOR[arch])

    bars = ax_gap.barh(y_pos, deltas, color=colors, alpha=0.80, height=0.55,
                       edgecolor='white', linewidth=0.5)

    for bar, delta, pval, arch in zip(bars, deltas, pvals, archs):
        stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
        ax_gap.text(delta + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{delta:.3f} {stars}",
                    va='center', ha='left', fontsize=9,
                    color=ARCH_COLOR[arch], fontweight='bold')

    ax_gap.set_yticks(y_pos)
    ax_gap.set_yticklabels(archs[::-1], fontsize=11)
    ax_gap.set_xlabel(r"$\Delta = \bar{f}^{\mathrm{FIM}}_{\mathrm{low}}"
                      r" - \bar{f}^{\mathrm{rand}}_{\mathrm{low}}$",
                      fontsize=11)
    ax_gap.set_title("(b) Gap $\\Delta$\n(higher = more aligned)",
                     fontsize=11)
    ax_gap.set_xlim(0, max(deltas) * 1.35)
    ax_gap.axvline(0, color='#888888', lw=1.0)

    # 总标题
    fig.suptitle(
        "FIM principal subspace preferentially captures low-frequency physical modes\n"
        r"(Cylinder dataset · perturbation $\varepsilon = 10^{-3}$ · K=20 directions each)",
        fontsize=12, y=1.02)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = f"{OUTPUT_DIR}/rq1_multi_arch_vs_random.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("RQ1 Multi-Architecture FIM vs Random Baseline")
    log.info(f"  K={K}  N_FIM={N_FIM}  N_TEST={N_TEST}  ε={EPSILON}")
    log.info("=" * 60)

    # 公共数据集 & normalizer（所有架构共用）
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer

    common   = dict(dataset_name="cylinder", dataset_root=DATASET_ROOT)
    train_ds = CylinderHFDataset(mode="train", dataset_type="numerical", **common)
    val_ds   = CylinderHFDataset(mode="val",   dataset_type="real",      **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)
    log.info(f"  train={len(train_ds)}  val={len(val_ds)}")

    results = {}
    for arch in ARCH_CONFIGS:
        log.info(f"\n>>> Architecture: {arch}")
        fl_top, fl_rand = run_arch(arch, train_ds, val_ds, normalizer)
        results[arch] = (fl_top, fl_rand)
        _, pval = mannwhitneyu(fl_top, fl_rand, alternative='greater')
        log.info(f"  [{arch}]  FIM={fl_top.mean():.4f}±{fl_top.std():.4f}  "
                 f"Rand={fl_rand.mean():.4f}±{fl_rand.std():.4f}  "
                 f"Δ={fl_top.mean()-fl_rand.mean():+.4f}  p={pval:.2e}")

    # 汇总
    log.info("\n" + "="*60)
    log.info("SUMMARY")
    log.info(f"{'Arch':<12} {'FIM mean':>10} {'Rand mean':>10} {'Δ':>8} {'p':>10}")
    for arch, (fl_top, fl_rand) in results.items():
        _, pval = mannwhitneyu(fl_top, fl_rand, alternative='greater')
        log.info(f"{arch:<12} {fl_top.mean():>10.4f} {fl_rand.mean():>10.4f} "
                 f"{fl_top.mean()-fl_rand.mean():>+8.4f} {pval:>10.2e}")

    log.info("\nPlotting …")
    make_figure(results)
    log.info("Done.")


if __name__ == "__main__":
    main()
