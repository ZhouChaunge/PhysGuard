#!/usr/bin/env python3
"""
Direct Perturbation Proof Figure
=================================
Directly proves that FIM principal subspace encodes low-frequency
physics modes by visualizing the output perturbation Δy:

  - Perturb θ along FIM rank-1  → Δy is spatially smooth (coherent physics)
  - Perturb θ along FIM rank-20 → Δy is less smooth (transitioning)
  - Perturb θ along random dir  → Δy is noisy (incoherent high-freq)

Figure layout (2 rows):
  Row 1: Δy spatial heatmaps for rank-1 / rank-20 / random (same color scale)
  Row 2: Radial PSD comparison curves (log scale)

Run:
  cd RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python \\
      -u ../scripts/rq1_direct_proof.py
"""

import os, sys, logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.join(SCRIPT_DIR, "..", "RealPDEBench")
sys.path.insert(0, REPO_DIR)

DATASET_ROOT = "./data/realpdebench/"
CKPT_PATH    = ("./results/001-cylinder"
                "/fno/fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth")
OUTPUT_DIR   = "./figures"
CACHE_PATH   = "/tmp/rq1_direct_proof_cache.npz"

# FNO Layer 2 — strongest layer (ρ = −0.93 from prior analysis)
TARGET_PARAM = "spectral_convs.1.weights1"

N_FIM     = 60      # gradient samples for FIM estimation
N_TEST    = 40      # test samples for Δy computation
K_EIGEN   = 20      # how many eigenvectors to extract
K_SHOW    = [1, 10, 20]   # FIM ranks to include in PSD (1-indexed)
K_SPATIAL = [1, 20]       # FIM ranks shown as spatial fields
N_RANDOM  = 5       # number of random directions (averaged for PSD stability)
EPSILON   = 1e-3    # perturbation step size
FWD_BATCH = 8       # forward-pass batch size

T_SHOW  = 10   # time step for spatial field display
CH_SHOW = 0    # channel index: 0 = u-velocity (stream-wise)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  DATA & MODEL
# ═══════════════════════════════════════════════════════════════════

def build_all():
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer
    from realpdebench.model.load_model import load_model

    common   = dict(dataset_name="cylinder", dataset_root=DATASET_ROOT)
    train_ds = CylinderHFDataset(mode="train", dataset_type="numerical", **common)
    val_ds   = CylinderHFDataset(mode="val",   dataset_type="real",      **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)

    model = load_model(train_ds, device=DEVICE,
                       model_name="fno", modes1=4, modes2=12, modes3=16,
                       n_layers=4, width=64)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    raw   = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return train_ds, val_ds, normalizer, model


# ═══════════════════════════════════════════════════════════════════
#  FIM EIGENVECTORS
# ═══════════════════════════════════════════════════════════════════

def collect_grads(model, loader, normalizer, n):
    grads = []
    pbar  = tqdm(total=n, desc="FIM grads")
    cnt   = 0
    for x, y in loader:
        if cnt >= n:
            break
        x, _ = normalizer.preprocess(x, y)
        x = x[:1].to(DEVICE)
        model.zero_grad()
        model(x).pow(2).mean().backward()
        for pname, p in model.named_parameters():
            if pname != TARGET_PARAM:
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


def top_k_eigenvectors(grads, k):
    J  = torch.stack(grads, dim=0)          # [N_FIM, param_dim]
    G  = J @ J.T                             # [N_FIM, N_FIM]
    L, V = np.linalg.eigh(G.numpy().astype(np.float64))
    L = np.maximum(L[::-1], 0.0)
    V = V[:, ::-1]
    k_act = min(k, int((L > 1e-12).sum()))
    U = []
    for j in range(k_act):
        vj = torch.tensor(V[:, j], dtype=torch.float32)
        uj = J.T @ vj / (L[j] ** 0.5)
        uj /= (uj.norm() + 1e-12)
        U.append(uj)
    return torch.stack(U, dim=1), L[:k_act]   # [param_dim, k_act]


# ═══════════════════════════════════════════════════════════════════
#  BASE OUTPUTS
# ═══════════════════════════════════════════════════════════════════

def get_base_outputs(model, loader, normalizer, n):
    xs, ys = [], []
    cnt = 0
    with torch.no_grad():
        for x, y in loader:
            if cnt >= n:
                break
            x, _ = normalizer.preprocess(x, y)
            rem  = min(n - cnt, x.shape[0])
            x    = x[:rem]
            xs.append(x.cpu())
            ys.append(model(x.to(DEVICE)).cpu())
            cnt += rem
    return torch.cat(xs), torch.cat(ys)


# ═══════════════════════════════════════════════════════════════════
#  2-D SPATIAL FFT → RADIAL PSD
# ═══════════════════════════════════════════════════════════════════

def compute_2d_psd(dy: torch.Tensor):
    """
    dy: [B, T, H, W, C]
    Returns:
      psd  : 1-D array of length `half`  (ring-averaged, mean over B,T,C)
      half : int
    """
    B, T, H, W, C = dy.shape
    half = min(H // 2, W // 2)

    # Average over time → [B, H, W, C]
    dy_avg = dy.float().mean(dim=1)

    # 2-D FFT over spatial dims
    F = torch.fft.fftn(dy_avg, dim=[1, 2])            # [B, H, W, C]
    P = F.abs() ** 2
    P_half = P[:, :H // 2, :W // 2, :]                # take positive quadrant only

    # Ring-average
    ii = torch.arange(H // 2).float()
    jj = torch.arange(W // 2).float()
    gi, gj = torch.meshgrid(ii, jj, indexing='ij')
    radial = torch.floor(torch.sqrt(gi ** 2 + gj ** 2)).long()
    valid  = radial < half

    r_flat  = radial[valid]
    # sum across B and C
    P_bc = P_half.permute(0, 3, 1, 2).reshape(B * C, H // 2, W // 2)
    P_sum = P_bc.sum(dim=0)                            # [H//2, W//2]
    P_valid = P_sum[valid]

    psd = torch.zeros(half)
    psd.scatter_add_(0, r_flat, P_valid)

    cnt = torch.zeros(half, dtype=torch.long)
    cnt.scatter_add_(0, r_flat, torch.ones_like(r_flat))
    cnt = cnt.clamp(min=1)
    psd = (psd / (cnt.float() * B * C)).numpy()

    return psd, half


def f_low(psd, half):
    k_th = max(1, round(half / 3))
    return float(psd[:k_th].sum() / (psd.sum() + 1e-15))


# ═══════════════════════════════════════════════════════════════════
#  PERTURBATION ALONG A DIRECTION VECTOR
# ═══════════════════════════════════════════════════════════════════

def perturb_direction(model, u_vec, x_test, y_base, eps=EPSILON):
    """
    Perturb TARGET_PARAM by eps * u_vec, run forward pass, restore.
    Returns (delta_y [B,T,H,W,C] CPU, psd [half], half).
    """
    for pname, p in model.named_parameters():
        if pname != TARGET_PARAM:
            continue
        if p.is_complex():
            d  = p.numel()
            re = u_vec[:d].reshape(p.shape).to(p.device)
            im = u_vec[d:].reshape(p.shape).to(p.device)
            delta = torch.complex(re, im).to(dtype=p.dtype)
        else:
            delta = u_vec.reshape(p.shape).to(p.device, dtype=p.dtype)

        with torch.no_grad():
            p.data.add_(eps * delta)

        ys = []
        with torch.no_grad():
            for st in range(0, x_test.shape[0], FWD_BATCH):
                ys.append(model(x_test[st:st + FWD_BATCH].to(DEVICE)).cpu())
        y_pert = torch.cat(ys)

        with torch.no_grad():
            p.data.sub_(eps * delta)

        dy = y_pert - y_base
        psd, half = compute_2d_psd(dy)
        return dy, psd, half

    raise ValueError(f"Parameter '{TARGET_PARAM}' not found in model.")


# ═══════════════════════════════════════════════════════════════════
#  COMPUTE ALL RESULTS
# ═══════════════════════════════════════════════════════════════════

def compute_results(model, U, x_test, y_base):
    """
    Returns a flat dict with all arrays needed for both figure and cache.
    """
    B, T, H, W, C = y_base.shape
    results = {}

    # ── FIM ranks ──
    for rank in K_SHOW:
        if rank > U.shape[1]:
            log.warning(f"Rank {rank} exceeds available eigenvectors, skipping.")
            continue
        u_j = U[:, rank - 1].to(DEVICE)
        log.info(f"Perturbing along FIM rank {rank} …")
        dy, psd, half = perturb_direction(model, u_j, x_test, y_base)
        fl = f_low(psd, half)
        log.info(f"  f_low = {fl:.4f}")
        results[f'psd_{rank}']      = psd
        results[f'f_low_{rank}']    = fl
        results[f'half']            = half
        # spatial slice for display
        if rank in K_SPATIAL:
            results[f'dy_slice_{rank}'] = (
                dy[0, T_SHOW, :, :, CH_SHOW].numpy())      # [H, W]

    # ── Random directions ──
    param_dim = U.shape[0]
    log.info(f"Perturbing along {N_RANDOM} random direction(s) …")
    psd_rand_list = []
    dy_rand_list  = []
    for seed in range(N_RANDOM):
        torch.manual_seed(seed + 100)
        r = torch.randn(param_dim)
        r /= (r.norm() + 1e-12)
        dy, psd_r, half = perturb_direction(model, r.to(DEVICE), x_test, y_base)
        psd_rand_list.append(psd_r)
        if seed == 0:
            # keep first random Δy for spatial display
            dy_rand_list.append(dy[0, T_SHOW, :, :, CH_SHOW].numpy())
    psd_random = np.mean(psd_rand_list, axis=0)
    fl_random  = f_low(psd_random, half)
    log.info(f"  Random f_low = {fl_random:.4f}")
    results['psd_random']      = psd_random
    results['f_low_random']    = fl_random
    results['dy_slice_random'] = dy_rand_list[0]

    return results


# ═══════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════

def make_figure(results):
    """
    results: flat dict with keys psd_{rank}, f_low_{rank}, dy_slice_{rank},
             psd_random, f_low_random, dy_slice_random, half.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    half  = int(results['half'])
    k_th  = max(1, round(half / 3))
    ks    = np.arange(half)

    # ── Color palette ──
    c_rank1  = '#1565C0'   # deep blue
    c_rank10 = '#FF8F00'   # amber
    c_rank20 = '#2E7D32'   # dark green
    c_random = '#C62828'   # deep red

    # ── Spatial panels: rank-1, rank-20, random ──
    spatial_keys = [(K_SPATIAL[0], f'FIM rank {K_SPATIAL[0]}\n(top eigenvector)', c_rank1),
                    (K_SPATIAL[1], f'FIM rank {K_SPATIAL[1]}\n(lower importance)', c_rank20),
                    ('random',     'Random direction',                              c_random)]

    # Layout: Row 1 = 3 spatial panels, Row 2 = 1 wide PSD panel
    fig = plt.figure(figsize=(12, 6.2))
    gs_top = gridspec.GridSpec(1, 3, left=0.05, right=0.97, top=0.88,
                               bottom=0.47, wspace=0.18)
    gs_bot = gridspec.GridSpec(1, 1, left=0.09, right=0.97, top=0.40,
                               bottom=0.10)

    # ── Row 1: spatial Δy fields ──
    # Compute shared color range from both FIM-spatial panels (FAIRER comparison)
    # but cap at 80th percentile to avoid outliers dominating
    spatial_fields = []
    for (key_rank, _, _) in spatial_keys:
        k_str = key_rank if key_rank == 'random' else key_rank
        arr = results[f'dy_slice_{key_rank}']
        spatial_fields.append(arr)

    # Global vmax: 90th percentile of absolute values (suppress outlier pixels)
    vmax = np.percentile([np.abs(f).max() for f in spatial_fields], 90)
    if vmax == 0:
        vmax = 1.0

    axs_top = []
    for col, (key_rank, title, border_color) in enumerate(spatial_keys):
        ax = fig.add_subplot(gs_top[col])
        field = results[f'dy_slice_{key_rank}']
        fl_val = (results[f'f_low_{key_rank}']
                  if key_rank == 'random' else results[f'f_low_{key_rank}'])
        im = ax.imshow(field.T, origin='lower', aspect='auto',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       interpolation='bilinear')
        # Title with colored box
        ax.set_title(title, fontsize=11, pad=5, fontweight='bold', color=border_color)
        ax.set_xlabel("stream-wise $x$", fontsize=9)
        if col == 0:
            ax.set_ylabel("cross-stream $y$", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color); spine.set_linewidth(2.2)
        # f_low annotation
        fl_color = '#003399' if fl_val > 0.75 else '#990000'
        ax.text(0.97, 0.04, f"$f_{{\\rm low}} = {fl_val:.3f}$",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, fontweight='bold', color=fl_color,
                bbox=dict(fc='white', ec='silver', alpha=0.90, pad=2.5, boxstyle='round'))
        # Panel label
        label = ['(a)', '(b)', '(c)'][col]
        ax.text(-0.06, 1.10, label, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')
        axs_top.append(ax)

    # Shared colorbar for spatial panels
    plt.colorbar(im, ax=axs_top, fraction=0.015, pad=0.01,
                 label="$\\Delta y$  (u-velocity, normalized)")

    # ── Row 2: Radial PSD comparison ──
    ax_psd = fig.add_subplot(gs_bot[0])

    psd_info = [
        (1,        c_rank1,  f'FIM rank 1  ($f_{{\\rm low}}={results["f_low_1"]:.3f}$)',   '-',  2.2),
        (10,       c_rank10, f'FIM rank 10  ($f_{{\\rm low}}={results["f_low_10"]:.3f}$)', '--', 1.8),
        (20,       c_rank20, f'FIM rank 20  ($f_{{\\rm low}}={results["f_low_20"]:.3f}$)', '--', 1.8),
        ('random', c_random, f'Random dir.  ($f_{{\\rm low}}={results["f_low_random"]:.3f}$)', '-.',1.8),
    ]

    for (key_rank, color, label, ls, lw) in psd_info:
        psd = results[f'psd_{key_rank}']
        psd_norm = psd / (psd.max() + 1e-15)
        ax_psd.plot(ks, psd_norm, color=color, lw=lw, ls=ls, label=label)

    # Low-freq band
    ax_psd.axvspan(0, k_th - 0.5, alpha=0.08, color='steelblue',
                   label=f'Low-freq band  ($k < {k_th}$)')
    ax_psd.axvline(k_th - 0.5, color='steelblue', lw=1.2, ls=':', alpha=0.7)

    ax_psd.set_yscale('log')
    ax_psd.set_xlabel("Radial wavenumber  $k$", fontsize=11)
    ax_psd.set_ylabel("Normalized PSD", fontsize=11)
    ax_psd.set_title("(d)  Radial power spectral density of output perturbation  $\\Delta y$",
                     fontsize=11, pad=5)
    ax_psd.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
    ax_psd.set_xlim(0, min(half - 1, half))
    ax_psd.set_ylim(bottom=1e-6)
    ax_psd.grid(axis='y', alpha=0.3)

    # ── Super title ──
    fig.suptitle(
        "FIM top eigenvectors concentrate output sensitivity in low-frequency physics modes\n"
        f"(FNO · Cylinder wake · Layer 2 · $\\varepsilon = {EPSILON}$)",
        fontsize=11.5, y=0.96)

    # ── Save ──
    for ext in ("pdf", "png"):
        path = os.path.join(OUTPUT_DIR, f"rq1_direct_proof.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)
    log.info("Figure complete.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 62)
    log.info("RQ1 Direct Perturbation Proof")
    log.info(f"  Target : {TARGET_PARAM}")
    log.info(f"  N_FIM={N_FIM}  N_TEST={N_TEST}  K_SHOW={K_SHOW}")
    log.info(f"  ε={EPSILON}  T_show={T_SHOW}  ch={CH_SHOW}")
    log.info("=" * 62)

    # ── Load cache if available ──
    if os.path.exists(CACHE_PATH):
        log.info(f"Cache found: {CACHE_PATH}  — loading …")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        results = {k: cache[k] for k in cache.files}
        # scalar items stored as 0-d arrays → convert
        for k in list(results.keys()):
            if results[k].ndim == 0:
                results[k] = float(results[k])
        log.info("Cache loaded. Skipping computation.")
        make_figure(results)
        return

    # ── Build model & data ──
    log.info("Loading model and datasets …")
    train_ds, val_ds, normalizer, model = build_all()
    log.info(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    loader_fim  = DataLoader(train_ds, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    loader_test = DataLoader(val_ds,   batch_size=FWD_BATCH, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── FIM eigenvectors ──
    log.info(f"Collecting {N_FIM} FIM gradient samples …")
    grads = collect_grads(model, loader_fim, normalizer, N_FIM)
    U, eigvals = top_k_eigenvectors(grads, K_EIGEN)
    k_act = U.shape[1]
    log.info(f"  eigvals: λ₁={eigvals[0]:.3e}  λ{k_act}={eigvals[-1]:.3e}")
    log.info(f"  U shape: {tuple(U.shape)}")

    # ── Base outputs ──
    log.info(f"Computing base outputs for {N_TEST} test samples …")
    x_test, y_base = get_base_outputs(model, loader_test, normalizer, N_TEST)
    log.info(f"  y_base: {tuple(y_base.shape)}")

    # ── Perturbation experiments ──
    results = compute_results(model, U, x_test, y_base)

    # ── Cache ──
    np.savez(CACHE_PATH, **results)
    log.info(f"Cache saved → {CACHE_PATH}")

    # ── Figure ──
    log.info("Generating figure …")
    make_figure(results)
    log.info("Done.")


if __name__ == "__main__":
    main()
