#!/usr/bin/env python3
"""
FIM Low-Frequency Alignment — FNO on Multiple Datasets
=======================================================
Runs the same intuitive FIM heatmap + spatial-field visualization
as rq1_fim_intuitive_viz.py (which handled FNO/Cylinder), but for:
  - controlled_cylinder  (FNO, spectral_convs.1.weights1, modes2=12)
  - combustion           (FNO, spectral_convs.1.weights1, modes2=16)

Output per dataset:
  figures/rq1_fno_{dataset_tag}_intuitive.pdf/.png
  figures/rq1_fno_{dataset_tag}_intuitive_data.npz

Run:
  cd ./RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python \\
      -u ../scripts/rq1_fno_multi_dataset_intuitive_viz.py 2>&1 | tee /tmp/rq1_fno_multi_ds.log
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
CKPT_BASE    = "./results"
OUTPUT_DIR   = "./figures"

TARGET_PARAM = "spectral_convs.1.weights1"   # strongest FIM layer (ρ=-0.93 on cylinder)

N_FIM     = 50
N_TEST    = 30
K_EIGEN   = 20
EPSILON   = 1e-3
FWD_BATCH = 8

SHOW_RANKS = [1, 7, 14, 20]   # representative spread for spatial panel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Dataset / checkpoint configurations ──────────────────────────
DATASET_CONFIGS = {
    "controlled_cylinder": {
        "tag":          "ctrl_cylinder",
        "display_name": "Controlled Cylinder",
        "dataset_class":"ControlledCylinderHFDataset",
        "modes1": 4, "modes2": 12, "modes3": 16, "n_layers": 4, "width": 64,
        "ckpt": (f"{CKPT_BASE}/002-control_cylinder/fno"
                 "/fno_control_pretrained/2026-03-15_02-36-15/model_4000.pth"),
    },
    "combustion": {
        "tag":          "combustion",
        "display_name": "Combustion",
        "dataset_class":"CombustionHFDataset",
        "modes1": 4, "modes2": 16, "modes3": 16, "n_layers": 4, "width": 64,
        "ckpt": (f"{CKPT_BASE}/003-combustion/fno"
                 "/fno_combustion_pretrained/2026-03-20_11-54-28/model_2000.pth"),
    },
}


# ═══════════════════════════════════════════════════════════════════
#  FFT: 3-D radial power spectrum (same as rq1_fim_intuitive_viz.py)
# ═══════════════════════════════════════════════════════════════════

_grid_cache = {}

def _build_grid(T, H, W, device):
    half = min(T // 2, H // 2, W // 2)
    ii   = torch.arange(T // 2, device=device).float()
    jj   = torch.arange(H // 2, device=device).float()
    kk   = torch.arange(W // 2, device=device).float()
    gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing='ij')
    radial = torch.floor(torch.sqrt(gi**2 + gj**2 + gk**2)).long()
    valid  = radial < half
    return radial, valid, half


def radial_power_spectrum(delta_y: torch.Tensor):
    """
    delta_y: [B, T, H, W, C]
    Returns (err_F, half): 1-D radial power spectrum array, bin count.
    """
    T, H, W = delta_y.shape[1], delta_y.shape[2], delta_y.shape[3]
    key = (T, H, W, delta_y.device.type)
    if key not in _grid_cache:
        _grid_cache[key] = _build_grid(T, H, W, delta_y.device)
    radial, valid, half = _grid_cache[key]

    D          = torch.fft.fftn(delta_y.float(), dim=[1, 2, 3])
    power      = D.abs() ** 2
    power_half = power[:, :T//2, :H//2, :W//2, :]

    B, C       = power_half.shape[0], power_half.shape[-1]
    pfv        = power_half.permute(0, 4, 1, 2, 3).reshape(B * C, -1)
    v_flat     = valid.reshape(-1)
    r_flat     = radial.reshape(-1)[v_flat]
    pfv_valid  = pfv[:, v_flat]
    pfv_sum    = pfv_valid.sum(dim=0)
    err_F      = torch.zeros(half, device=delta_y.device)
    err_F.scatter_add_(0, r_flat, pfv_sum)
    err_F     /= (B * C)
    return err_F.cpu().numpy(), half


# ═══════════════════════════════════════════════════════════════════
#  DATA & MODEL
# ═══════════════════════════════════════════════════════════════════

def build_datasets(ds_name, ds_class_name):
    from realpdebench.data.fluid_hf_dataset import (
        ControlledCylinderHFDataset,
    )
    from realpdebench.data.combustion_hf_dataset import CombustionHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer

    DS_CLS = {"ControlledCylinderHFDataset": ControlledCylinderHFDataset,
               "CombustionHFDataset": CombustionHFDataset}[ds_class_name]

    common   = dict(dataset_name=ds_name, dataset_root=DATASET_ROOT)
    train_ds = DS_CLS(mode="train", dataset_type="numerical", **common)
    val_ds   = DS_CLS(mode="val",   dataset_type="real",      **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)
    return train_ds, val_ds, normalizer


def load_fno(train_ds, cfg):
    from realpdebench.model.load_model import load_model
    model = load_model(train_ds, device=DEVICE, model_name="fno",
                       modes1=cfg["modes1"], modes2=cfg["modes2"],
                       modes3=cfg["modes3"], n_layers=cfg["n_layers"],
                       width=cfg["width"])
    ckpt = torch.load(cfg["ckpt"], map_location=DEVICE)
    if isinstance(ckpt, dict):
        raw = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        raw = ckpt
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


# ═══════════════════════════════════════════════════════════════════
#  FIM EIGENVECTORS
# ═══════════════════════════════════════════════════════════════════

def collect_grads(model, loader, normalizer, n):
    grads = []
    pbar  = tqdm(total=n, desc=f"  FIM grads [{TARGET_PARAM.split('.')[-1]}]", leave=False)
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
            xs.append(x.cpu())
            ys.append(model(x.to(DEVICE)).cpu())
            cnt += rem
    return torch.cat(xs), torch.cat(ys)


def perturb(model, u_j, x_test, y_base):
    for pname, p in model.named_parameters():
        if pname != TARGET_PARAM: continue
        if p.is_complex():
            d  = p.numel()
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

def make_figure(spectra_matrix, delta_ys_dict, half, display_name, rho, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import uniform_filter1d

    K = spectra_matrix.shape[0]

    # normalise each spectrum so max=1
    spec_norm = spectra_matrix.copy()
    for j in range(K):
        mx = spec_norm[j].max()
        if mx > 0: spec_norm[j] /= mx

    spec_smooth = uniform_filter1d(spec_norm, size=3, axis=1)

    n_show = len(SHOW_RANKS)
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, n_show], wspace=0.35,
                            left=0.06, right=0.97, top=0.88, bottom=0.12)

    # ── Panel (a): heatmap ──
    ax_heat = fig.add_subplot(gs[0])
    k_show  = min(30, half)
    img     = spec_smooth[:, :k_show].T   # [k_show, K]

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

    ax_heat.text(0.03, 0.08, "Low-freq\n(physics)", transform=ax_heat.transAxes,
                 fontsize=8.5, color='white', va='bottom',
                 bbox=dict(fc='#1f77b4', ec='none', alpha=0.75, pad=2))
    ax_heat.text(0.03, 0.62, "High-freq\n(noise)", transform=ax_heat.transAxes,
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
        dy = delta_ys_dict[j][sample_idx, t_idx, :, :, ch_idx].numpy()
        all_fields.append(dy)

    vmax = max(np.abs(f).max() for f in all_fields)
    if vmax == 0: vmax = 1.0

    for col, (j, field) in enumerate(zip(SHOW_RANKS, all_fields)):
        ax = fig.add_subplot(gs_b[col])
        ax.imshow(field.T, origin='lower', aspect='auto',
                  cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  interpolation='bilinear')
        ax.set_title(f"j = {j}", fontsize=10, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Space (y)", fontsize=9)

    fig.text(0.63, 0.93,
             "(b)  Output perturbation field $\\Delta y$  (first channel, t=5)",
             ha='center', va='center', fontsize=11)

    fig.suptitle(
        f"FIM principal subspace preferentially captures low-frequency physics\n"
        f"(FNO  ·  {display_name}  ·  {TARGET_PARAM}  ·  Spearman $\\rho = {rho:+.2f}$)",
        fontsize=11.5, y=1.01)

    for ext in ["pdf", "png"]:
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  PER-DATASET ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def run_dataset(ds_name, cfg):
    tag          = cfg["tag"]
    display_name = cfg["display_name"]
    out_base     = f"{OUTPUT_DIR}/rq1_fno_{tag}_intuitive"
    npz_path     = f"{out_base}_data.npz"

    log.info(f"\n{'='*60}")
    log.info(f"  FNO  |  {display_name}  ({ds_name})")
    log.info(f"{'='*60}")

    # ── Skip if already done ──
    if os.path.exists(npz_path):
        log.info(f"  Cache found → {npz_path}  (skipping recompute)")
        d = np.load(npz_path, allow_pickle=True)
        log.info(f"  f_low @j=1={d['f_low'][0]:.3f}  "
                 f"@j=20={d['f_low'][-1]:.3f}  "
                 f"ρ={float(d['rho']):.3f}  p={float(d['pval']):.3e}")
        return

    # ── Load data ──
    log.info("  Loading dataset …")
    train_ds, val_ds, normalizer = build_datasets(ds_name, cfg["dataset_class"])
    log.info(f"  train: {len(train_ds)}  val: {len(val_ds)}")
    x0, y0 = train_ds[0]
    log.info(f"  sample shape: x={tuple(x0.shape)}  y={tuple(y0.shape)}")

    loader_fim  = DataLoader(train_ds, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    loader_test = DataLoader(val_ds, batch_size=FWD_BATCH, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── Load FNO ──
    log.info(f"  Loading FNO (modes1={cfg['modes1']}, modes2={cfg['modes2']}, "
             f"modes3={cfg['modes3']}, n_layers={cfg['n_layers']}, width={cfg['width']}) …")
    model = load_fno(train_ds, cfg)

    # Verify target param exists
    param_dict = dict(model.named_parameters())
    if TARGET_PARAM not in param_dict:
        avail = [n for n in param_dict if "spectral" in n]
        raise RuntimeError(f"{TARGET_PARAM} not found. Available spectral params: {avail}")
    p_numel = param_dict[TARGET_PARAM].numel()
    log.info(f"  Target: {TARGET_PARAM}  ({p_numel} elements, "
             f"complex={param_dict[TARGET_PARAM].is_complex()})")

    # ── FIM eigenvectors ──
    grads = collect_grads(model, loader_fim, normalizer, N_FIM)
    U, eigvals = top_k_eigenvectors(grads, K_EIGEN)
    k_act = U.shape[1]
    log.info(f"  k_act={k_act}  λ_1={eigvals[0]:.3e}  λ_{k_act}={eigvals[-1]:.3e}")

    # ── Base outputs ──
    x_test, y_base = get_base_outputs(model, loader_test, normalizer, N_TEST)
    log.info(f"  x_test: {tuple(x_test.shape)}")

    # ── Perturb along each eigenvector ──
    spectra  = []
    delta_ys = {}
    for j_idx in tqdm(range(k_act), desc="  sweep FIM ranks", leave=True):
        u_j = U[:, j_idx]
        dy, spec, half = perturb(model, u_j, x_test, y_base)
        spectra.append(spec)
        j1 = j_idx + 1
        if j1 in SHOW_RANKS:
            delta_ys[j1] = dy

    spectra_matrix = np.stack(spectra, axis=0)   # [k_act, half]
    log.info(f"  spectra: {spectra_matrix.shape}  half={half}")

    # ── Spearman ρ on f_low ──
    iLow = int(round(half / 3))
    f_low = spectra_matrix[:, :iLow].sum(axis=1) / (spectra_matrix.sum(axis=1) + 1e-30)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(np.arange(k_act), f_low)
    log.info(f"  f_low @j=1: {f_low[0]:.4f}  @j={k_act}: {f_low[-1]:.4f}")
    log.info(f"  Spearman ρ = {rho:+.3f}  (p={pval:.3e})")

    # ── Save data ──
    np.savez(npz_path, spectra=spectra_matrix, f_low=f_low,
             rho=rho, pval=pval, half=half)
    log.info(f"  Data → {npz_path}")

    # ── Figure ──
    make_figure(spectra_matrix, delta_ys, half, display_name, rho, out_base)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("FNO FIM Heatmap  —  Multi-Dataset")
    log.info(f"  Target param : {TARGET_PARAM}")
    log.info(f"  N_FIM={N_FIM}  N_TEST={N_TEST}  K_EIGEN={K_EIGEN}  ε={EPSILON}")
    log.info(f"  Datasets     : {list(DATASET_CONFIGS.keys())}")
    log.info("=" * 60)

    for ds_name, cfg in DATASET_CONFIGS.items():
        run_dataset(ds_name, cfg)
        torch.cuda.empty_cache()

    log.info("\nAll datasets done.")


if __name__ == "__main__":
    main()
