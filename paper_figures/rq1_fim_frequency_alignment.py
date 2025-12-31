#!/usr/bin/env python3
"""
RQ1: Does the FIM physics subspace correspond to low-frequency structures?

Experiment: Perturbation Sensitivity Analysis
─────────────────────────────────────────────
For each FIM eigenvector u_j (ranked by eigenvalue λ_j, j=1…N):
  1. Perturb θ* → θ* + ε·u_j  (only the parameters of one layer at a time)
  2. Forward on N_test simulation inputs → ŷ_pert
  3. Compute Δy = ŷ_pert − ŷ_orig
  4. Decompose ||FFT(Δy)||² into low / mid / high frequency bands
  5. Record low-frequency fraction = low / (low+mid+high)

If FIM principal directions correspond to low-frequency physics (the paper's
core assumption), low_fraction should be highest for j=1 and decrease with j.

Output
──────
  figures/rq1_fim_frequency_alignment.pdf   (and .png)

Run (from RealPDEBench/ directory):
  CUDA_VISIBLE_DEVICES=0 python ../scripts/rq1_fim_frequency_alignment.py

Settings are hard-coded at the top; adjust if needed.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# ─────────────────────────── CONFIG ───────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.join(SCRIPT_DIR, "..", "RealPDEBench")
sys.path.insert(0, REPO_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = 0                        # single GPU

CHECKPOINT = (
    "./results/001-cylinder/fno/"
    "fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth"
)
DATASET_ROOT = "./data/realpdebench/"
OUTPUT_DIR   = "./figures"

# FIM estimation samples (from simulation train split)
N_FIM   = 50
# Validation samples used for output perturbation measurement
N_TEST  = 50
# Number of eigenvectors to analyse per layer
K_EIGEN = 30
# Perturbation scale (multiplied by unit eigenvector)
EPSILON = 1e-3
# Batch size used during perturbation forward pass
FWD_BATCH = 8

# FNO model config (matches fno_nullspace.yaml)
FNO_CFG = dict(
    model_name="fno",
    modes1=4, modes2=12, modes3=16,
    n_layers=4, width=64,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─────────────────────────── HELPERS ───────────────────────────

def _build_radial_grid(T, H, W, device):
    """
    Return (radial, valid_mask, half) for the half-space [T//2, H//2, W//2].
    radial: long tensor [T//2, H//2, W//2] of floor(sqrt(i²+j²+k²))
    valid_mask: bool tensor [T//2, H//2, W//2], True where radial < half
    """
    half = min(T // 2, H // 2, W // 2)
    ii = torch.arange(T // 2, device=device).float()
    jj = torch.arange(H // 2, device=device).float()
    kk = torch.arange(W // 2, device=device).float()
    gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing='ij')
    radial = torch.floor(torch.sqrt(gi**2 + gj**2 + gk**2)).long()
    valid = radial < half
    return radial, valid, half


def freq_band_energy(delta_y: torch.Tensor, _cache: dict = {}):
    """
    Vectorized radial-wavenumber binning.
    delta_y: [B, T, H, W, C]
    Returns (e_low, e_mid, e_high) as Python floats.
    """
    T, H, W = delta_y.shape[1], delta_y.shape[2], delta_y.shape[3]
    key = (T, H, W, delta_y.device.type)
    if key not in _cache:
        _cache[key] = _build_radial_grid(T, H, W, delta_y.device)
    radial, valid, half = _cache[key]

    # FFT over spatial+temporal dims; take power in half-space
    D = torch.fft.fftn(delta_y.float(), dim=[1, 2, 3])          # [B,T,H,W,C]
    power = D.abs() ** 2                                          # [B,T,H,W,C]
    power_half = power[:, :T//2, :H//2, :W//2, :]               # [B,T/2,H/2,W/2,C]

    # Flatten spatial dims, mask, scatter into radial bins
    B, C = power_half.shape[0], power_half.shape[-1]
    pfv = power_half.permute(0, 4, 1, 2, 3).reshape(B * C, -1)  # [B*C, T/2*H/2*W/2]
    v_flat = valid.reshape(-1)                                    # [T/2*H/2*W/2]
    r_flat = radial.reshape(-1)[v_flat]                           # [M]
    pfv_valid = pfv[:, v_flat]                                    # [B*C, M]

    # bin-sum across all (b,c) pairs, then normalise
    pfv_sum = pfv_valid.sum(dim=0)                                # [M]
    err_F = torch.zeros(half, device=delta_y.device)
    err_F.scatter_add_(0, r_flat, pfv_sum)
    err_F /= (B * C)

    iLow  = int(round(half / 3))
    iHigh = int(round(half * 2 / 3))
    e_low  = err_F[:iLow].sum().item()
    e_mid  = err_F[iLow:iHigh].sum().item()
    e_high = err_F[iHigh:].sum().item()
    return e_low, e_mid, e_high


def get_flat_params(model):
    """Return a concatenated flat view of all leaf parameters (real representation)."""
    return torch.cat([
        p.data.reshape(-1) if not p.is_complex()
        else torch.cat([p.data.reshape(-1).real, p.data.reshape(-1).imag])
        for p in model.parameters() if p.requires_grad
    ])


def named_params_list(model):
    """Return list of (name, param) for params with requires_grad."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


# ─────────────────────────── DATA LOADING ───────────────────────────

def build_datasets():
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer

    common = dict(dataset_name="cylinder", dataset_root=DATASET_ROOT)
    sim_train = CylinderHFDataset(mode="train", dataset_type="numerical", **common)
    # Numerical sim data has no val split → use real val for output perturbation test
    # (We only use input x for forward pass; perturbation comparison is architecture-level)
    sim_val   = CylinderHFDataset(mode="val",   dataset_type="real", **common)
    # Build normalizer from simulation training data
    normalizer = GaussianNormalizer(sim_train, device=DEVICE)
    return sim_train, sim_val, normalizer


# ─────────────────────────── MODEL LOADING ───────────────────────────

def build_model(train_dataset, normalizer):
    from realpdebench.model.load_model import load_model
    inp, tgt = train_dataset[0]
    cfg = dict(**FNO_CFG)
    model = load_model(train_dataset, device=DEVICE, **cfg)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    if "model_state_dict" in ckpt:
        raw = ckpt["model_state_dict"]
    else:
        raw = ckpt
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    log.info(f"Loaded FNO checkpoint from {CHECKPOINT}")
    return model


# ─────────────────────────── FIM COMPUTATION ───────────────────────────

def collect_gradients_per_layer(model, dataloader, normalizer, n_samples,
                                target_names=None):
    """
    Collect per-sample gradients for selected layers.
    target_names: if given, only accumulate gradients for these parameter names.
    Returns: grad_accum: dict { param_name -> list of 1-D float16 tensors }
    """
    if target_names is None:
        target_names = {name for name, _ in named_params_list(model)}
    else:
        target_names = set(target_names)
    grad_accum = {name: [] for name in target_names}
    n_collected = 0

    for batch in tqdm(dataloader, desc="  Collecting gradients"):
        if n_collected >= n_samples:
            break
        x, y = batch
        x, y = normalizer.preprocess(x, y)

        B = x.shape[0]
        for i in range(B):
            if n_collected >= n_samples:
                break
            model.zero_grad()
            xi = x[i:i+1]
            yi = y[i:i+1]
            loss = model.train_loss(xi, yi).mean()
            loss.backward()

            for name, param in named_params_list(model):
                if param.grad is None or name not in target_names:
                    continue
                g = param.grad.detach().reshape(-1)
                if g.is_complex():
                    g = torch.cat([g.real, g.imag])
                grad_accum[name].append(g.cpu().half())
            n_collected += 1

    log.info(f"  Collected {n_collected} gradient samples.")
    return grad_accum


def compute_eigenvectors(grad_list, k_max):
    """
    Given a list of N gradient vectors (1-D, float16, CPU),
    compute the top-k FIM eigenvectors via the Gram trick.
    Returns:
      eigenvalues : [N] float32 tensor (descending)
      eigenvectors: [d, N] float32 tensor  (columns = FIM eigenvectors)
    """
    G = torch.stack(grad_list, dim=0).float()   # [N, d]
    N, d = G.shape
    K = G @ G.t()                               # [N, N]
    # Symmetric eigendecomposition (descending by default after sort)
    vals, vecs = torch.linalg.eigh(K)           # vals ascending
    vals = vals.flip(0).clamp(min=0)            # descending, remove numerical negatives
    vecs = vecs.flip(1)                         # [N, N] columns descending

    # Map back to parameter space: u_j = G^T v_j  (then normalise)
    U = G.t() @ vecs                            # [d, N]
    norms = U.norm(dim=0, keepdim=True).clamp(min=1e-12)
    U = U / norms                               # unit eigenvectors

    k = min(k_max, N)
    return vals[:k], U[:, :k]                   # [k],  [d, k]


# ─────────────────────────── PERTURBATION MEASUREMENT ───────────────────────────

def compute_base_outputs(model, loader, normalizer, n_test):
    """Forward the first n_test samples in batches, return (x_norm, base_outputs)."""
    all_x, all_y_base = [], []
    n = 0
    with torch.no_grad():
        for x, y in loader:
            if n >= n_test:
                break
            x, _ = normalizer.preprocess(x, y)
            remaining = n_test - n
            x = x[:remaining]
            xd = x.to(DEVICE)
            y_pred = model(xd).cpu()
            all_x.append(x.cpu())
            all_y_base.append(y_pred)
            n += x.shape[0]
    return torch.cat(all_x, dim=0), torch.cat(all_y_base, dim=0)


def perturb_and_measure(model, param_name, param_ref, u_j, x_test, y_base, eps=EPSILON):
    """
    Temporarily perturb one parameter along unit vector u_j,
    run forward on x_test, compute freq band power of (ŷ_pert - ŷ_base).
    Returns (e_low, e_mid, e_high).
    """
    # Map (name) → (param object)
    for n, p in named_params_list(model):
        if n == param_name:
            target_param = p
            break

    # Build perturbation in original shape.
    # grad was stored as cat([g.real, g.imag], dim=0), so u_j is twice the complex numel.
    if target_param.is_complex():
        d_cplx = target_param.numel()          # number of complex elements
        delta_real = u_j[:d_cplx].reshape(target_param.shape).to(device=target_param.device)
        delta_imag = u_j[d_cplx:].reshape(target_param.shape).to(device=target_param.device)
        delta = torch.complex(delta_real, delta_imag).to(dtype=target_param.dtype)
    else:
        delta = u_j.reshape(target_param.shape).to(device=target_param.device,
                                                     dtype=target_param.dtype)

    # Apply perturbation
    with torch.no_grad():
        target_param.data.add_(eps * delta)

    # Forward pass in batches
    y_pert_list = []
    with torch.no_grad():
        for start in range(0, x_test.shape[0], FWD_BATCH):
            xb = x_test[start:start+FWD_BATCH].to(DEVICE)
            yb = model(xb).cpu()
            y_pert_list.append(yb)
    y_pert = torch.cat(y_pert_list, dim=0)

    # Restore parameter
    with torch.no_grad():
        target_param.data.sub_(eps * delta)

    # Frequency band energy of perturbation
    delta_y = y_pert - y_base    # [N_test, T, H, W, C]
    return freq_band_energy(delta_y)


# ─────────────────────────── WEIGHT-SPACE ANALYSIS ───────────────────────────

def weight_space_frequency_analysis(U, param_shape, is_complex):
    """
    For each FIM eigenvector u_j, compute the fraction of ||u_j||² that falls
    in each spectral-mode frequency band.

    Works by reshaping u_j to the original parameter layout and computing
    per-mode energy, then binning by radial wavenumber r=floor(sqrt(m1²+m2²+m3²)).

    Args:
        U:           [d_real, K] float32  FIM eigenvectors (unit norm cols)
        param_shape: the original shape of the complex parameter tensor
                     (e.g. [64,64,4,12,16] for SpectralConv weights)
        is_complex:  True if the parameter was complex (u encodes real+imag)

    Returns:
        low_fracs, mid_fracs, high_fracs : [K] numpy arrays
    """
    K = U.shape[1]
    # For complex weight [C_in, C_out, m1, m2, m3]: d_complex = prod(param_shape)
    # d_real = 2 * d_complex → U[:d_complex,:] = real part, U[d_complex:,:] = imag part
    if not is_complex:
        # Real weight: just reshape u_j → param_shape, compute per-element energy
        modes_shape = param_shape[-3:]  # last 3 dims are (m1, m2, m3)
        channel_shape = param_shape[:-3]
        d_tot = U.shape[0]
        E_U = U.pow(2)                          # [d_real, K]
        # Reshape to [..., m1, m2, m3, K] then sum over channel dims
        E_reshaped = E_U.reshape(*param_shape, K)
        E_modes = E_reshaped.reshape(-1, *modes_shape, K).sum(dim=0)  # [m1,m2,m3,K]
    else:
        d_cplx = 1
        for s in param_shape:
            d_cplx *= s                         # total complex elements
        # First d_cplx rows = real part; next d_cplx = imag part
        E_real = U[:d_cplx, :].pow(2)           # [d_cplx, K]
        E_imag = U[d_cplx:, :].pow(2)           # [d_cplx, K]
        E_U = E_real + E_imag                   # [d_cplx, K]
        modes_shape = param_shape[-3:]
        E_reshaped = E_U.reshape(*param_shape, K)
        E_modes = E_reshaped.reshape(-1, *modes_shape, K).sum(dim=0)  # [m1,m2,m3,K]

    # Build radial wavenumber grid for the mode space
    m1, m2, m3 = [torch.arange(s).float() for s in modes_shape]
    gm1, gm2, gm3 = torch.meshgrid(m1, m2, m3, indexing='ij')
    radial = torch.floor(torch.sqrt(gm1**2 + gm2**2 + gm3**2)).long()
    max_r = radial.max().item()
    half  = max_r // 2 if max_r >= 4 else max_r
    iLow  = max(1, round(half / 3))
    iHigh = max(iLow + 1, round(half * 2 / 3))

    log.info(f"    Weight mode-space: max_r={max_r}, half={half}, "
             f"iLow={iLow}, iHigh={iHigh}, modes={modes_shape}")

    low_fracs  = np.zeros(K)
    mid_fracs  = np.zeros(K)
    high_fracs = np.zeros(K)
    r_flat = radial.reshape(-1)   # [m1*m2*m3]

    for j in range(K):
        e_j = E_modes[..., j].reshape(-1)          # [m1*m2*m3]
        e_low  = e_j[r_flat < iLow].sum().item()
        e_mid  = e_j[(r_flat >= iLow) & (r_flat < iHigh)].sum().item()
        e_high = e_j[r_flat >= iHigh].sum().item()
        total  = e_low + e_mid + e_high + 1e-30
        low_fracs[j]  = e_low  / total
        mid_fracs[j]  = e_mid  / total
        high_fracs[j] = e_high / total

    return low_fracs, mid_fracs, high_fracs


# ─────────────────────────── MAIN ANALYSIS ───────────────────────────

def analyse_layer(model, layer_name, grad_list, x_test, y_base, k_max=K_EIGEN):
    """
    Return arrays of shape [k_max] for low/mid/high frequency fractions
    and for eigenvalues, for a given layer.
    """
    log.info(f"  Computing Gram SVD for {layer_name} …")
    eigenvalues, U = compute_eigenvectors(grad_list, k_max)
    k = eigenvalues.shape[0]

    # Find the corresponding parameter object once
    param_ref = None
    for n, p in named_params_list(model):
        if n == layer_name:
            param_ref = p
            break

    low_fracs  = np.zeros(k)
    mid_fracs  = np.zeros(k)
    high_fracs = np.zeros(k)

    for j in tqdm(range(k), desc=f"    Eigenvec rank ({layer_name[-30:]})", leave=False):
        u_j = U[:, j].to(DEVICE)
        e_low, e_mid, e_high = perturb_and_measure(
            model, layer_name, param_ref, u_j, x_test, y_base
        )
        total = e_low + e_mid + e_high + 1e-30
        low_fracs[j]  = e_low  / total
        mid_fracs[j]  = e_mid  / total
        high_fracs[j] = e_high / total

    return eigenvalues.numpy(), low_fracs, mid_fracs, high_fracs


# ─────────────────────────── PLOTTING ───────────────────────────

def make_figure(layer_results):
    """
    layer_results: dict { layer_short_name ->
        (eigenvalues, ws_low, ws_mid, ws_high, os_low, os_mid, os_high) }
    ws_* : weight-space frequency fractions  (always available)
    os_* : output-space frequency fractions  (may be None if skipped)

    Layout: 3 rows × N_layers columns
      Row 0: eigenvalue spectrum
      Row 1: weight-space freq fractions    (FNO-specific)
      Row 2: output-perturbation freq fracs (generic)
    """
    layers = list(layer_results.keys())
    n = len(layers)
    has_output = any(layer_results[l][4] is not None for l in layers)
    nrows = 3 if has_output else 2
    fig, axes = plt.subplots(nrows, n, figsize=(4.5 * n, 3.5 * nrows),
                             constrained_layout=True)
    if n == 1:
        axes = axes[:, np.newaxis]

    colors = {"low": "#2166ac", "mid": "#74add1", "high": "#fdae61"}

    for col, lname in enumerate(layers):
        evs, ws_low, ws_mid, ws_high, os_low, os_mid, os_high = layer_results[lname]
        ranks = np.arange(1, len(evs) + 1)

        # ── Row 0: eigenvalue spectrum ──
        ax0 = axes[0, col]
        ax0.semilogy(ranks, evs / evs[0], color="#555", lw=1.5, marker="o",
                     ms=3, markevery=max(1, len(ranks)//10))
        ax0.set_xlabel("Rank $j$", fontsize=9)
        ax0.set_ylabel("$\\lambda_j / \\lambda_1$" if col == 0 else "", fontsize=9)
        ax0.set_title(lname, fontsize=9, fontweight="bold")
        ax0.grid(True, which="both", linestyle="--", alpha=0.4)
        ax0.set_xlim(1, len(ranks))

        # ── Row 1: weight-space fracs ──
        ax1 = axes[1, col]
        ax1.stackplot(ranks, ws_low, ws_mid, ws_high,
                      labels=["Low-$k$", "Mid-$k$", "High-$k$"],
                      colors=[colors["low"], colors["mid"], colors["high"]],
                      alpha=0.85)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel("Rank $j$", fontsize=9)
        ax1.set_ylabel("Fraction of $\\|u_j\\|^2$\n(weight mode space)"
                       if col == 0 else "", fontsize=9)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax1.set_xlim(1, len(ranks))
        if col == 0:
            ax1.legend(loc="lower right", fontsize=7.5, framealpha=0.7)

        # ── Row 2: output-space fracs ──
        if has_output and os_low is not None:
            ax2 = axes[2, col]
            os_ranks = np.arange(1, len(os_low) + 1)
            ax2.stackplot(os_ranks, os_low, os_mid, os_high,
                          labels=["Low-$f$", "Mid-$f$", "High-$f$"],
                          colors=[colors["low"], colors["mid"], colors["high"]],
                          alpha=0.85)
            ax2.axhline(1/3, color="k", lw=0.8, ls=":", label="Uniform (1/3)")
            ax2.set_ylim(0, 1)
            ax2.set_xlabel("Rank $j$", fontsize=9)
            ax2.set_ylabel("Fraction of $\\|\\Delta y\\|^2$\n(output freq space)"
                           if col == 0 else "", fontsize=9)
            ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax2.set_xlim(1, len(os_ranks))
            if col == 0:
                ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.7)

    # Row labels
    row_labels = ["(a) Eigenvalue decay", "(b) Weight-space freq fraction",
                  "(c) Output-space freq fraction"]
    for r in range(nrows):
        axes[r, 0].set_ylabel(f"{row_labels[r]}\n{axes[r, 0].get_ylabel()}", fontsize=9)

    fig.suptitle(
        "RQ1: Do FIM principal eigenvectors correspond to low-frequency physical structures?\n"
        "FNO pretrained on Cylinder Flow  |  "
        "If hypothesis holds: low-freq fraction should be highest at $j{=}1$ and decrease with rank.",
        fontsize=9,
    )
    return fig


# ─────────────────────────── ENTRY POINT ───────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if DEVICE == "cuda":
        torch.cuda.set_device(GPU_ID)

    log.info("Loading datasets …")
    sim_train, sim_val, normalizer = build_datasets()

    log.info("Building model …")
    model = build_model(sim_train, normalizer)

    # ── Select layers to analyse ──
    # Focus on SpectralConv3d weights1 of each FNO layer
    LAYERS_TO_ANALYSE = [
        name for name, _ in named_params_list(model)
        if "spectral_convs" in name and "weights1" in name
    ]
    if not LAYERS_TO_ANALYSE:
        # Fallback: take parameters with most elements
        params_by_size = sorted(
            [(n, p.numel()) for n, p in named_params_list(model)],
            key=lambda x: x[1], reverse=True)
        LAYERS_TO_ANALYSE = [n for n, _ in params_by_size[:4]]

    log.info(f"Layers to analyse: {LAYERS_TO_ANALYSE}")

    # ── Collect FIM gradients (only for target layers to save memory) ──
    fim_loader = DataLoader(
        Subset(sim_train, list(range(min(N_FIM * 2, len(sim_train))))),
        batch_size=4, shuffle=False, num_workers=4, pin_memory=True,
    )
    log.info(f"Collecting gradients for {N_FIM} samples (target layers only) …")
    grad_accum = collect_gradients_per_layer(
        model, fim_loader, normalizer, N_FIM,
        target_names=LAYERS_TO_ANALYSE
    )

    # ── Precompute base outputs for output-perturbation analysis ──
    test_loader = DataLoader(
        Subset(sim_val, list(range(min(N_TEST, len(sim_val))))),
        batch_size=FWD_BATCH, shuffle=False, num_workers=2,
    )
    log.info("Computing base outputs ŷ_orig …")
    x_test, y_base = compute_base_outputs(model, test_loader, normalizer, N_TEST)
    log.info(f"  x_test {tuple(x_test.shape)}, y_base {tuple(y_base.shape)}")

    # ── Per-layer analysis ──
    layer_results = {}   # lname → (evs, ws_low, ws_mid, ws_high, os_low, os_mid, os_high)

    for lname in LAYERS_TO_ANALYSE:
        grad_list = grad_accum.get(lname)
        if not grad_list:
            log.warning(f"No gradients for {lname}, skipping.")
            continue

        # Parameter metadata
        target_param = None
        for n, p in named_params_list(model):
            if n == lname:
                target_param = p
                break

        short = lname  # keep full name for clarity; trim if needed
        # e.g. spectral_convs.0.weights1 → conv0.w1
        parts = lname.split(".")
        short = f"Layer {parts[-2] if len(parts)>=2 else lname}.{parts[-1]}"

        log.info(f"\n── {lname} (N={len(grad_list)}, d_grad={grad_list[0].numel()}) ──")

        # Eigenvectors
        log.info("  Computing eigenvectors via Gram trick …")
        eigenvalues, U = compute_eigenvectors(grad_list, K_EIGEN)
        log.info(f"  λ_1={eigenvalues[0]:.3e}, λ_{len(eigenvalues)}={eigenvalues[-1]:.3e}")

        # ── Weight-space analysis ──
        log.info("  Running weight-space frequency analysis …")
        ws_low, ws_mid, ws_high = weight_space_frequency_analysis(
            U.cpu(), target_param.shape, target_param.is_complex()
        )
        log.info(f"  WS freq @j=1:  low={ws_low[0]:.3f} mid={ws_mid[0]:.3f} high={ws_high[0]:.3f}")
        log.info(f"  WS freq @j={len(eigenvalues)}: low={ws_low[-1]:.3f} mid={ws_mid[-1]:.3f} high={ws_high[-1]:.3f}")

        # ── Output-perturbation analysis ──
        log.info("  Running output-perturbation frequency analysis …")
        _, os_low, os_mid, os_high = analyse_layer(
            model, lname, grad_list, x_test, y_base, k_max=K_EIGEN
        )
        log.info(f"  OS freq @j=1:  low={os_low[0]:.3f} mid={os_mid[0]:.3f} high={os_high[0]:.3f}")
        log.info(f"  OS freq @j={len(eigenvalues)}: low={os_low[-1]:.3f} mid={os_mid[-1]:.3f} high={os_high[-1]:.3f}")

        layer_results[short] = (eigenvalues.numpy(),
                                ws_low, ws_mid, ws_high,
                                os_low, os_mid, os_high)

    # ── Summary: Pearson correlation ──
    log.info("\n── Summary ──")
    for lname, (evs, ws_low, ws_mid, ws_high, os_low, os_mid, os_high) in layer_results.items():
        ranks = np.arange(len(evs))
        r_ws = np.corrcoef(ranks, ws_low)[0, 1]
        r_os = np.corrcoef(ranks, os_low)[0, 1] if os_low is not None else float('nan')
        log.info(f"  {lname}: weight-space Pearson(rank, low_f) = {r_ws:.3f}  "
                 f"output-space = {r_os:.3f}")
        log.info(f"    (negative ⇒ low-freq fraction DECREASES with rank → hypothesis SUPPORTED)")

    # ── Save data ──
    npz_path = os.path.join(OUTPUT_DIR, "rq1_fim_frequency_data.npz")
    save_dict = {}
    for lname, (evs, ws_low, ws_mid, ws_high, os_low, os_mid, os_high) in layer_results.items():
        safe = lname.replace(".", "_").replace(" ", "_")
        save_dict[f"{safe}_eigenvalues"] = evs
        save_dict[f"{safe}_ws_low"]  = ws_low
        save_dict[f"{safe}_ws_mid"]  = ws_mid
        save_dict[f"{safe}_ws_high"] = ws_high
        if os_low is not None:
            save_dict[f"{safe}_os_low"]  = os_low
            save_dict[f"{safe}_os_mid"]  = os_mid
            save_dict[f"{safe}_os_high"] = os_high
    np.savez(npz_path, **save_dict)
    log.info(f"\nSaved data → {npz_path}")

    # ── Plot ──
    fig = make_figure(layer_results)
    for ext in ("pdf", "png"):
        out = os.path.join(OUTPUT_DIR, f"rq1_fim_frequency_alignment.{ext}")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        log.info(f"Saved figure → {out}")
    plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    main()
