#!/usr/bin/env python3
"""
FIM Low-Frequency Alignment — Intuitive Visualization for CNO & DeepONet
=========================================================================
Same two-panel figure as rq1_fim_intuitive_viz.py but for CNO and DeepONet,
which have NO frequency truncation, so the heatmap should show a much more
dramatic low→high frequency shift as j increases.

  (a) Frequency–rank heatmap: X=FIM rank j, Y=radial wavenumber k,
      Color = normalized spectral energy per eigenvector direction
  (b) Spatial Δy fields at j=1, 7, 14, 20

CNO best layer   : auto-selected top-1 by numel (from prev run: "convolution.weight", ρ=-0.687)
DeepONet best layer: branch.conv4.0.weight (256ch, ρ=-0.487, p=0.029)

Run:
  cd RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python \
      -u ../scripts/rq1_cno_deeponet_intuitive_viz.py 2>&1 | tee /tmp/rq1_cno_deeponet_intuitive.log
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
CKPT_BASE    = "./results/001-cylinder"
OUTPUT_DIR   = "./figures"

CKPT_CNO      = f"{CKPT_BASE}/cno/cno_cylinder_pretrained/2026-03-11_09-49-04/model_5000.pth"
CKPT_DEEPONET = f"{CKPT_BASE}/deeponet/deeponet_cylinder_pretrained/2026-03-12_11-16-03/model_5000.pth"

N_FIM     = 50
N_TEST    = 30
K_EIGEN   = 20
EPSILON   = 1e-3
FWD_BATCH = 8

SHOW_RANKS = [1, 7, 14, 20]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  FFT: full radial power spectrum
# ═══════════════════════════════════════════════════════════════════

_grid_cache = {}

def _build_grid(T, H, W, device):
    # Use only spatial dims H, W for richer frequency decomposition
    half = min(H // 2, W // 2)
    jj  = torch.arange(H // 2, device=device).float()
    kk  = torch.arange(W // 2, device=device).float()
    gj, gk = torch.meshgrid(jj, kk, indexing='ij')
    radial = torch.floor(torch.sqrt(gj**2 + gk**2)).long()
    valid  = radial < half
    return radial, valid, half

def radial_power_spectrum(delta_y: torch.Tensor):
    """
    delta_y: [B, T, H, W, C]
    Returns err_F: 1-D array of length `half`.
    Uses 2D spatial FFT (mean over time) for higher wavenumber resolution.
    """
    T, H, W = delta_y.shape[1], delta_y.shape[2], delta_y.shape[3]
    key = (H, W, delta_y.device.type)
    if key not in _grid_cache:
        _grid_cache[key] = _build_grid(T, H, W, delta_y.device)
    radial, valid, half = _grid_cache[key]

    # Average over time → [B, H, W, C], then 2D FFT
    dy_mean = delta_y.float().mean(dim=1)          # [B, H, W, C]
    D       = torch.fft.fftn(dy_mean, dim=[1, 2])  # [B, H, W, C]
    power   = D.abs() ** 2
    power_half = power[:, :H//2, :W//2, :]         # [B, H//2, W//2, C]

    B, C    = power_half.shape[0], power_half.shape[-1]
    pfv     = power_half.permute(0, 3, 1, 2).reshape(B * C, -1)  # [BC, H//2*W//2]
    v_flat  = valid.reshape(-1)
    r_flat  = radial.reshape(-1)[v_flat]
    pfv_valid = pfv[:, v_flat]
    pfv_sum   = pfv_valid.sum(dim=0)
    err_F     = torch.zeros(half, device=delta_y.device)
    err_F.scatter_add_(0, r_flat, pfv_sum)
    err_F /= (B * C)
    return err_F.cpu().numpy(), half


# ═══════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════

def build_datasets():
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer

    common   = dict(dataset_name="cylinder", dataset_root=DATASET_ROOT)
    train_ds = CylinderHFDataset(mode="train", dataset_type="numerical", **common)
    val_ds   = CylinderHFDataset(mode="val",   dataset_type="real",      **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)
    return train_ds, val_ds, normalizer


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        raw = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        raw = ckpt
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def select_top_param(model, min_numel=50_000):
    """Return the name of the largest real-valued weight matrix."""
    best_name, best_n = None, 0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.dim() < 2:
            continue
        low = name.lower()
        if any(t in low for t in ("bias", "norm", ".bn", "batch_norm")):
            continue
        n = p.numel()
        if n >= min_numel and n > best_n:
            best_n, best_name = n, name
    return best_name


# ═══════════════════════════════════════════════════════════════════
#  FIM EIGENVECTORS
# ═══════════════════════════════════════════════════════════════════

def collect_grads(model, loader, normalizer, param_name, n):
    grads = []
    pbar  = tqdm(total=n, desc=f"  FIM grads [{param_name.split('.')[-1]}]", leave=False)
    cnt   = 0
    for x, y in loader:
        if cnt >= n: break
        x, _ = normalizer.preprocess(x, y)
        x = x[:1].to(DEVICE)
        model.zero_grad()
        model(x).pow(2).mean().backward()
        for pname, p in model.named_parameters():
            if pname != param_name: continue
            if p.grad is None: break
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


def perturb(model, param_name, u_j, x_test, y_base):
    """Perturb param along u_j, return (delta_y, radial_spec, half)."""
    for pname, p in model.named_parameters():
        if pname != param_name: continue
        delta = u_j.reshape(p.shape).to(device=p.device, dtype=p.dtype)

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
    raise ValueError(f"{param_name} not found")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════

def make_figure(spectra_matrix, delta_ys_dict, half,
                arch_name, param_name, rho, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import uniform_filter1d
    from scipy.stats import spearmanr

    K = spectra_matrix.shape[0]

    # ── normalise each spectrum so max=1 ──
    spec_norm = spectra_matrix.copy()
    for j in range(K):
        mx = spec_norm[j].max()
        if mx > 0: spec_norm[j] /= mx

    spec_smooth = uniform_filter1d(spec_norm, size=2, axis=1)

    n_show = len(SHOW_RANKS)
    fig = plt.figure(figsize=(12, 4.8))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, n_show], wspace=0.35,
                            left=0.06, right=0.97, top=0.88, bottom=0.12)

    # ── Panel (a): heatmap ──
    ax_heat = fig.add_subplot(gs[0])
    k_show  = min(32, half)
    img     = spec_smooth[:, :k_show].T          # [k_show, K]

    im = ax_heat.imshow(img, aspect='auto', origin='lower',
                        extent=[0.5, K + 0.5, 0, k_show],
                        cmap='RdYlBu_r', vmin=0, vmax=1,
                        interpolation='bilinear')

    iLow = int(round(half / 3))
    ax_heat.axhline(iLow, color='white', lw=1.5, ls='--', alpha=0.8)

    ax_heat.set_xlabel("FIM eigenvector rank $j$  (1 = most important)", fontsize=11)
    ax_heat.set_ylabel("Radial wavenumber $k$", fontsize=11)
    ax_heat.set_title("(a) Spectral energy of output perturbation\n"
                      "per FIM direction", fontsize=11, pad=6)
    ax_heat.set_xticks([1, 5, 10, 15, 20])

    cb = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label("Norm. energy", fontsize=9)

    ax_heat.text(0.03, 0.06, "Low-freq\n(physics)", transform=ax_heat.transAxes,
                 fontsize=8.5, color='white', va='bottom',
                 bbox=dict(fc='#1f77b4', ec='none', alpha=0.75, pad=2))
    ax_heat.text(0.03, 0.60, "High-freq\n(noise)", transform=ax_heat.transAxes,
                 fontsize=8.5, color='white', va='bottom',
                 bbox=dict(fc='#d62728', ec='none', alpha=0.70, pad=2))

    # ── Panel (b): Δy spatial fields ──
    gs_b = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[1],
                                            wspace=0.08)
    sample_idx = 0
    t_idx      = 5
    ch_idx     = 0

    all_fields = []
    for j in SHOW_RANKS:
        dy = delta_ys_dict[j]
        f  = dy[sample_idx, t_idx, :, :, ch_idx].numpy()
        all_fields.append(f)
    vmax = max(abs(f).max() for f in all_fields) + 1e-30
    vmax = float(np.percentile([abs(f).max() for f in all_fields], 95))

    for idx, (j, field) in enumerate(zip(SHOW_RANKS, all_fields)):
        ax = fig.add_subplot(gs_b[idx])
        ax.imshow(field, origin='lower', aspect='auto',
                  cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  interpolation='nearest')
        ax.set_title(f"$j = {j}$", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if idx == 0:
            ax.set_ylabel("Space (y)", fontsize=9)

    fig.suptitle(
        f"FIM principal subspace preferentially captures low-frequency physics\n"
        f"({arch_name}  ·  {param_name}  ·  Spearman $\\rho = {rho:+.2f}$)",
        fontsize=11, y=0.99)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])

    # add shared label for panel (b)
    fig.text(0.68, 0.01,
             "(b)  Output perturbation field $\\Delta y$  (velocity channel, t=5)",
             ha='center', fontsize=10)

    for ext in ["pdf", "png"]:
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  PER-ARCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def run_arch(arch_name, model, param_name,
             loader_fim, loader_test, normalizer, label_rho,
             out_base):
    log.info(f"\n{'='*60}")
    log.info(f"  {arch_name}  |  {param_name}")
    log.info(f"{'='*60}")

    # 1. FIM eigenvectors
    grads = collect_grads(model, loader_fim, normalizer, param_name, N_FIM)
    U, eigvals = top_k_eigenvectors(grads, K_EIGEN)
    k_act = U.shape[1]
    log.info(f"  k_act={k_act}  λ_1={eigvals[0]:.3e}  λ_{k_act}={eigvals[-1]:.3e}")

    # 2. Base outputs
    x_test, y_base = get_base_outputs(model, loader_test, normalizer, N_TEST)
    log.info(f"  x_test shape: {tuple(x_test.shape)}")

    # 3. Perturb along each eigenvector
    spectra   = []
    delta_ys  = {}
    for j_idx in tqdm(range(k_act), desc=f"  sweep {arch_name}", leave=True):
        u_j = U[:, j_idx]
        dy, spec, half = perturb(model, param_name, u_j, x_test, y_base)
        spectra.append(spec)
        j1 = j_idx + 1
        if j1 in SHOW_RANKS:
            delta_ys[j1] = dy

    spectra_matrix = np.stack(spectra, axis=0)   # [K, half]
    log.info(f"  spectrum matrix: {spectra_matrix.shape}  half={half}")

    # 4. Spearman ρ using f_low
    iLow = int(round(half / 3))
    f_low = spectra_matrix[:, :iLow].sum(axis=1) / (spectra_matrix.sum(axis=1) + 1e-30)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(np.arange(k_act), f_low)
    log.info(f"  f_low @j=1: {f_low[0]:.3f}  @j={k_act}: {f_low[-1]:.3f}")
    log.info(f"  Spearman ρ = {rho:+.3f}  (p={pval:.3e})")

    # 5. Figure
    make_figure(spectra_matrix, delta_ys, half,
                arch_name, param_name, rho, out_base)

    # Save data
    np.savez(f"{out_base}_data.npz",
             spectra=spectra_matrix, f_low=f_low, rho=rho, pval=pval)
    log.info(f"  Data → {out_base}_data.npz")

    return rho, pval


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    from realpdebench.model.load_model import load_model

    log.info("=" * 60)
    log.info("CNO & DeepONet FIM Intuitive Visualization")
    log.info(f"  N_FIM={N_FIM}  N_TEST={N_TEST}  K_EIGEN={K_EIGEN}  ε={EPSILON}")
    log.info("=" * 60)

    log.info("Loading Cylinder dataset …")
    train_ds, val_ds, normalizer = build_datasets()
    log.info(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    loader_fim  = DataLoader(train_ds, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    loader_test = DataLoader(val_ds,   batch_size=FWD_BATCH, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ─── CNO ──────────────────────────────────────────────────────
    log.info("\nBuilding CNO model …")
    cno = load_model(train_ds, device=DEVICE, model_name="cno", N_layers=3)
    load_ckpt(cno, CKPT_CNO)
    cno.to(DEVICE).eval()

    cno_param = select_top_param(cno, min_numel=50_000)
    log.info(f"  CNO best param: {cno_param}  ({dict(cno.named_parameters())[cno_param].numel()} params)")

    run_arch("CNO", cno, cno_param,
             loader_fim, loader_test, normalizer, None,
             out_base=f"{OUTPUT_DIR}/rq1_cno_intuitive")

    # free GPU memory before DeepONet
    del cno
    torch.cuda.empty_cache()

    # ─── DeepONet ─────────────────────────────────────────────────
    log.info("\nBuilding DeepONet model …")
    don = load_model(train_ds, device=DEVICE,
                     model_name="deeponet", p=128, dropout_rate=0.1)
    load_ckpt(don, CKPT_DEEPONET)
    don.to(DEVICE).eval()

    # Use Conv3 (128ch): ρ=-0.502 — strongest among DeepONet layers
    don_param = "branch.conv3.0.weight"
    # Verify it exists; fall back to conv4 if not
    param_dict = dict(don.named_parameters())
    if don_param not in param_dict:
        don_param = "branch.conv4.0.weight"
        if don_param not in param_dict:
            # Auto-select
            don_param = select_top_param(don, min_numel=10_000)
    log.info(f"  DeepONet param: {don_param}  ({param_dict.get(don_param, torch.empty(0)).numel()} params)")

    run_arch("DeepONet", don, don_param,
             loader_fim, loader_test, normalizer, None,
             out_base=f"{OUTPUT_DIR}/rq1_deeponet_intuitive")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
