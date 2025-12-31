#!/usr/bin/env python3
"""
FIM Low-Frequency Alignment — Intuitive Visualization
======================================================
More intuitive than f_low vs j line plots.

Figure layout (2 panels):
  (a) Frequency–rank heatmap: X = FIM rank j, Y = radial wavenumber k (low→high),
      Color = normalized spectral energy. If hypothesis holds:
        left columns (low j, high importance) = warm at bottom (low k)
        right columns (high j, low importance) = energy spreads upward (high k)

  (b) Actual output perturbation Δy fields for j=1 vs j=20:
      Shows smooth large-scale change vs noisy fine-grained change visually.

Uses FNO Layer 2 (spectral_convs.1.weights1, Spearman ρ = -0.93 — strongest layer).

Run:
  cd RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python -u ../scripts/rq1_fim_intuitive_viz.py
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

# Use the strongest layer from previous run
TARGET_PARAM = "spectral_convs.1.weights1"   # ρ = -0.93

N_FIM     = 50
N_TEST    = 30
K_EIGEN   = 20
EPSILON   = 1e-3
FWD_BATCH = 8

# Which ranks j to show Δy spatial fields for (panel b)
SHOW_RANKS = [1, 7, 14, 20]   # representative spread

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  FFT: full radial power spectrum (NOT just 3 bands)
# ═══════════════════════════════════════════════════════════════════

_grid_cache = {}

def _build_grid(T, H, W, device):
    half = min(T // 2, H // 2, W // 2)
    ii  = torch.arange(T // 2, device=device).float()
    jj  = torch.arange(H // 2, device=device).float()
    kk  = torch.arange(W // 2, device=device).float()
    gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing='ij')
    radial = torch.floor(torch.sqrt(gi**2 + gj**2 + gk**2)).long()
    valid  = radial < half
    return radial, valid, half

def radial_power_spectrum(delta_y: torch.Tensor):
    """
    delta_y: [B, T, H, W, C]
    Returns err_F: 1-D array of length `half`, radial power spectrum.
    """
    T, H, W = delta_y.shape[1], delta_y.shape[2], delta_y.shape[3]
    key = (T, H, W, delta_y.device.type)
    if key not in _grid_cache:
        _grid_cache[key] = _build_grid(T, H, W, delta_y.device)
    radial, valid, half = _grid_cache[key]

    D     = torch.fft.fftn(delta_y.float(), dim=[1, 2, 3])
    power = D.abs() ** 2
    power_half = power[:, :T//2, :H//2, :W//2, :]

    B, C    = power_half.shape[0], power_half.shape[-1]
    pfv     = power_half.permute(0, 4, 1, 2, 3).reshape(B * C, -1)
    v_flat  = valid.reshape(-1)
    r_flat  = radial.reshape(-1)[v_flat]
    pfv_valid = pfv[:, v_flat]
    pfv_sum   = pfv_valid.sum(dim=0)
    err_F     = torch.zeros(half, device=delta_y.device)
    err_F.scatter_add_(0, r_flat, pfv_sum)
    err_F /= (B * C)
    return err_F.cpu().numpy(), half


# ═══════════════════════════════════════════════════════════════════
#  DATA & MODEL
# ═══════════════════════════════════════════════════════════════════

def build_all():
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer
    from realpdebench.model.load_model import load_model

    common = dict(dataset_name="cylinder", dataset_root=DATASET_ROOT)
    train_ds = CylinderHFDataset(mode="train", dataset_type="numerical", **common)
    val_ds   = CylinderHFDataset(mode="val",   dataset_type="real",      **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)

    model = load_model(train_ds, device=DEVICE,
                       model_name="fno", modes1=4, modes2=12, modes3=16,
                       n_layers=4, width=64)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    raw  = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()

    return train_ds, val_ds, normalizer, model


# ═══════════════════════════════════════════════════════════════════
#  FIM EIGENVECTORS
# ═══════════════════════════════════════════════════════════════════

def collect_grads(model, loader, normalizer, n):
    grads = []
    pbar  = tqdm(total=n, desc="  FIM grads", leave=False)
    cnt   = 0
    for x, y in loader:
        if cnt >= n: break
        x, _ = normalizer.preprocess(x, y)
        x = x[:1].to(DEVICE)
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
    return grads

def top_k_eigenvectors(grads, k):
    J  = torch.stack(grads, dim=0)
    G  = J @ J.T
    L, V = np.linalg.eigh(G.numpy().astype(np.float64))
    L = np.maximum(L[::-1], 0); V = V[:, ::-1]
    k_act = min(k, int((L > 1e-12).sum()))
    U = []
    for j in range(k_act):
        vj = torch.tensor(V[:, j], dtype=torch.float32)
        uj = J.T @ vj / (L[j] ** 0.5)
        uj /= (uj.norm() + 1e-12)
        U.append(uj)
    return torch.stack(U, dim=1), L[:k_act]


# ═══════════════════════════════════════════════════════════════════
#  PERTURBATION
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

def perturb(model, u_j, x_test, y_base):
    """Returns Δy and full radial power spectrum."""
    for pname, p in model.named_parameters():
        if pname != TARGET_PARAM: continue
        if p.is_complex():
            d = p.numel()
            re = u_j[:d].reshape(p.shape).to(p.device)
            im = u_j[d:].reshape(p.shape).to(p.device)
            delta = torch.complex(re, im).to(dtype=p.dtype)
        else:
            delta = u_j.reshape(p.shape).to(p.device, dtype=p.dtype)

        with torch.no_grad(): p.data.add_(EPSILON * delta)

        ys = []
        with torch.no_grad():
            for st in range(0, x_test.shape[0], FWD_BATCH):
                ys.append(model(x_test[st:st+FWD_BATCH].to(DEVICE)).cpu())
        y_pert = torch.cat(ys)

        with torch.no_grad(): p.data.sub_(EPSILON * delta)

        delta_y = y_pert - y_base
        spec, half = radial_power_spectrum(delta_y)
        return delta_y, spec, half
    raise ValueError(f"{TARGET_PARAM} not found")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════

def make_figure(spectra_matrix, delta_ys_dict, half):
    """
    spectra_matrix: [K, half] — radial power spectrum per eigenvector rank j
    delta_ys_dict:  {j: Δy tensor [N,T,H,W,C]} for SHOW_RANKS
    half: number of radial wavenumber bins
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from scipy.ndimage import uniform_filter1d

    K = spectra_matrix.shape[0]

    # ── normalise each spectrum so max=1 (highlight shape, not magnitude) ──
    spec_norm = spectra_matrix.copy()
    for j in range(K):
        mx = spec_norm[j].max()
        if mx > 0: spec_norm[j] /= mx

    # ── smooth along wavenumber axis for cleaner heatmap ──
    spec_smooth = uniform_filter1d(spec_norm, size=3, axis=1)

    # ── layout ──
    n_show = len(SHOW_RANKS)
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, n_show], wspace=0.35,
                            left=0.06, right=0.97, top=0.88, bottom=0.12)

    # ────────────────────────────────────────────────────────────────
    # Panel (a): heatmap  [wavenumber k × rank j]
    # ────────────────────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0])

    # clip to first 30 wavenumber bins (most informative range)
    k_show = min(30, half)
    img    = spec_smooth[:, :k_show].T          # [k_show, K]

    im = ax_heat.imshow(img, aspect='auto', origin='lower',
                        extent=[0.5, K + 0.5, 0, k_show],
                        cmap='RdYlBu_r', vmin=0, vmax=1,
                        interpolation='bilinear')

    # annotate low/high freq boundary
    iLow  = int(round(half / 3))
    ax_heat.axhline(iLow, color='white', lw=1.5, ls='--', alpha=0.8,
                    label=f'Low/mid boundary (k={iLow})')

    ax_heat.set_xlabel("FIM eigenvector rank $j$  (1 = most important)", fontsize=11)
    ax_heat.set_ylabel("Radial wavenumber $k$", fontsize=11)
    ax_heat.set_title("(a) Spectral energy of output perturbation\n"
                      "per FIM direction", fontsize=11, pad=6)
    ax_heat.set_xticks([1, 5, 10, 15, 20])

    # colorbar
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label("Norm. energy", fontsize=9)

    # annotate low / high freq regions
    ax_heat.text(0.03, 0.08, "Low-freq\n(physics)", transform=ax_heat.transAxes,
                 fontsize=8.5, color='white', va='bottom',
                 bbox=dict(fc='#1f77b4', ec='none', alpha=0.75, pad=2))
    ax_heat.text(0.03, 0.62, "High-freq\n(noise)", transform=ax_heat.transAxes,
                 fontsize=8.5, color='white', va='bottom',
                 bbox=dict(fc='#d62728', ec='none', alpha=0.70, pad=2))

    # ────────────────────────────────────────────────────────────────
    # Panel (b): Δy spatial fields for selected ranks
    # ────────────────────────────────────────────────────────────────
    gs_b = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[1],
                                            wspace=0.08)

    # pick one sample and one time slice for display
    sample_idx = 0
    t_idx      = 5          # middle-ish time step
    ch_idx     = 0          # first output channel (e.g. velocity x)

    # compute common vmax across all selected ranks for fair comparison
    all_fields = []
    for j in SHOW_RANKS:
        dy = delta_ys_dict[j][sample_idx, t_idx, :, :, ch_idx].numpy()
        all_fields.append(dy)
    vmax_global = max(np.abs(f).max() for f in all_fields)
    if vmax_global == 0: vmax_global = 1.0

    for col, (j, field) in enumerate(zip(SHOW_RANKS, all_fields)):
        ax = fig.add_subplot(gs_b[col])
        ax.imshow(field.T, origin='lower', aspect='auto',
                  cmap='RdBu_r',
                  vmin=-vmax_global, vmax=vmax_global,
                  interpolation='bilinear')
        ax.set_title(f"j = {j}", fontsize=10, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Space (y)", fontsize=9)

    # group title for panel b
    fig.text(0.63, 0.93,
             "(b)  Output perturbation field $\\Delta y$  (velocity channel, t=5)",
             ha='center', va='center', fontsize=11)

    # overall figure title
    fig.suptitle("FIM principal subspace preferentially captures low-frequency physics\n"
                 f"(FNO · Layer 2 · {TARGET_PARAM}  ·  Spearman ρ = −0.93)",
                 fontsize=11.5, y=1.01)

    for ext in ["pdf", "png"]:
        path = os.path.join(OUTPUT_DIR, f"rq1_fim_intuitive.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("FIM Intuitive Visualization")
    log.info(f"  Target: {TARGET_PARAM}")
    log.info(f"  N_FIM={N_FIM}  N_TEST={N_TEST}  K_EIGEN={K_EIGEN}")
    log.info("=" * 60)

    log.info("Loading data & model …")
    train_ds, val_ds, normalizer, model = build_all()
    log.info(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    loader_fim  = DataLoader(train_ds, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    loader_test = DataLoader(val_ds,   batch_size=FWD_BATCH, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── FIM eigenvectors ──
    log.info(f"Collecting {N_FIM} FIM gradients …")
    grads = collect_grads(model, loader_fim, normalizer, N_FIM)
    U, eigvals = top_k_eigenvectors(grads, K_EIGEN)
    k_act = U.shape[1]
    log.info(f"  λ_1={eigvals[0]:.3e}  λ_{k_act}={eigvals[-1]:.3e}")

    # ── base outputs ──
    log.info("Computing base outputs …")
    x_test, y_base = get_base_outputs(model, loader_test, normalizer, N_TEST)
    log.info(f"  x_test: {tuple(x_test.shape)}")

    # ── sweep: collect full radial spectra ──
    spectra_list  = []
    delta_ys_dict = {}   # {j: Δy tensor}
    half_stored   = None

    log.info(f"Sweeping {k_act} eigenvectors …")
    for j_idx in tqdm(range(k_act), desc="  sweep"):
        u_j  = U[:, j_idx].to(DEVICE)
        dy, spec, half = perturb(model, u_j, x_test, y_base)
        spectra_list.append(spec)
        if half_stored is None: half_stored = half
        rank = j_idx + 1   # 1-indexed
        if rank in SHOW_RANKS:
            delta_ys_dict[rank] = dy   # [N,T,H,W,C] CPU tensor

    spectra_matrix = np.stack(spectra_list, axis=0)   # [K, half]
    log.info(f"  spectrum matrix: {spectra_matrix.shape}")

    # ── save data ──
    npz_path = os.path.join(OUTPUT_DIR, "rq1_fim_intuitive_data.npz")
    np.savez(npz_path, spectra_matrix=spectra_matrix, half=half_stored,
             show_ranks=np.array(SHOW_RANKS))
    log.info(f"Data saved → {npz_path}")

    # ── figure ──
    log.info("Generating figure …")
    make_figure(spectra_matrix, delta_ys_dict, half_stored)
    log.info("Done.")


if __name__ == "__main__":
    main()
