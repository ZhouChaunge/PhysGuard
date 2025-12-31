#!/usr/bin/env python3
"""
FNO FIM Low-Frequency Alignment — Theory Figure for PhysGuard
==============================================================
Demonstrates that FIM principal subspace preferentially captures
low-frequency physical modes, providing theoretical justification
for PhysGuard's protection of these directions.

Experiment:
  For each of FNO's 4 spectral conv layers (weights1):
    1. Estimate FIM via N_FIM per-sample gradients (Gram trick)
    2. For each top-K eigenvector u_j:
         θ ← θ* + ε·u_j  →  forward on N_TEST inputs  →  Δy
         f_low^(j) = Σ‖FFT(Δy)|_{low}‖² / Σ‖FFT(Δy)‖²
    3. Spearman ρ(j, f_low^(j))  (negative = top FIM dirs = low freq)

Output: figures/rq1_fno_fim_theory.pdf + .png

Run:
  cd RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python \
      -u ../scripts/rq1_fno_fim_theory.py
"""

import os, sys, logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# ─── paths ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.join(SCRIPT_DIR, "..", "RealPDEBench")
sys.path.insert(0, REPO_DIR)

DATASET_ROOT = "./data/realpdebench/"
CKPT_PATH    = ("./results/001-cylinder"
                "/fno/fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth")
OUTPUT_DIR   = "./figures"

# ─── hyper-params ───────────────────────────────────────────────────
N_FIM     = 50        # gradient samples for FIM estimation (more = stabler)
N_TEST    = 50        # validation samples for output-perturbation
K_EIGEN   = 20        # top-K FIM eigenvectors to sweep
EPSILON   = 1e-3      # perturbation magnitude ε
FWD_BATCH = 8         # forward batch size

# FNO config (cylinder pretrained)
FNO_KWARGS = dict(model_name="fno", modes1=4, modes2=12, modes3=16, n_layers=4, width=64)

# Target params: weights1 for each of the 4 spectral conv layers
TARGET_PARAMS = [f"spectral_convs.{i}.weights1" for i in range(4)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  FFT UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _build_radial_grid(T, H, W, device):
    half = min(T // 2, H // 2, W // 2)
    ii  = torch.arange(T // 2, device=device).float()
    jj  = torch.arange(H // 2, device=device).float()
    kk  = torch.arange(W // 2, device=device).float()
    gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing='ij')
    radial = torch.floor(torch.sqrt(gi**2 + gj**2 + gk**2)).long()
    valid  = radial < half
    return radial, valid, half

_radial_cache = {}

def freq_band_energy(delta_y: torch.Tensor):
    """
    delta_y: [B, T, H, W, C]
    Returns (e_low, e_mid, e_high) as floats.
    """
    T, H, W = delta_y.shape[1], delta_y.shape[2], delta_y.shape[3]
    key = (T, H, W, delta_y.device.type)
    if key not in _radial_cache:
        _radial_cache[key] = _build_radial_grid(T, H, W, delta_y.device)
    radial, valid, half = _radial_cache[key]

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

    iLow  = int(round(half / 3))
    iHigh = int(round(half * 2 / 3))
    e_low  = err_F[:iLow].sum().item()
    e_mid  = err_F[iLow:iHigh].sum().item()
    e_high = err_F[iHigh:].sum().item()
    return e_low, e_mid, e_high


def low_frac(e_low, e_mid, e_high):
    return e_low / (e_low + e_mid + e_high + 1e-30)


# ═══════════════════════════════════════════════════════════════════
#  DATA & MODEL
# ═══════════════════════════════════════════════════════════════════

def build_datasets():
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer

    common = dict(dataset_name="cylinder", dataset_root=DATASET_ROOT)
    train_ds = CylinderHFDataset(mode="train", dataset_type="numerical", **common)
    val_ds   = CylinderHFDataset(mode="val",   dataset_type="real",      **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)
    return train_ds, val_ds, normalizer


def build_fno(train_ds):
    from realpdebench.model.load_model import load_model
    return load_model(train_ds, device=DEVICE, **FNO_KWARGS)


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]
    else:
        raw = ckpt
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    log.info(f"  Loaded checkpoint: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════
#  FIM ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def collect_fim_gradients(model, loader, normalizer, param_name, n_samples):
    """
    Collect per-sample FIM gradient vectors for a single (complex) parameter.
    Returns list of 1-D float32 tensors (length = 2×numel for complex params).
    """
    grads = []
    n = 0
    pbar = tqdm(total=n_samples, desc=f"    FIM grads [{param_name.split('.')[-2:]}]",
                leave=False)
    for x, y in loader:
        if n >= n_samples:
            break
        x, _ = normalizer.preprocess(x, y)
        x = x[:1].to(DEVICE)

        model.zero_grad()
        y_hat = model(x)
        loss  = y_hat.pow(2).mean()
        loss.backward()

        for pname, p in model.named_parameters():
            if pname != param_name:
                continue
            g = p.grad
            if g is None:
                break
            if g.is_complex():
                gv = torch.cat([g.real.reshape(-1), g.imag.reshape(-1)]).float().cpu()
            else:
                gv = g.reshape(-1).float().cpu()
            grads.append(gv)
            break

        n += 1
        pbar.update(1)
    pbar.close()
    return grads


def top_k_eigenvectors(grads, k):
    """
    Gram trick: G = J·Jᵀ ∈ ℝ^{n×n}, n=len(grads).
    FIM ≈ Jᵀ·J. Top-k eigenvectors of FIM = top-k right singular vectors of J.
    Returns (U, eigenvalues) — U: [d, k], each column is a unit eigenvector.
    """
    J = torch.stack(grads, dim=0)          # [n, d]
    G = J @ J.T                             # [n, n]  — small Gram matrix
    G_np = G.numpy().astype(np.float64)
    L, V = np.linalg.eigh(G_np)            # ascending order
    L = L[::-1]; V = V[:, ::-1]            # descending
    L = np.maximum(L, 0)
    k_act = min(k, (L > 1e-12).sum())
    if k_act == 0:
        raise RuntimeError("All FIM eigenvalues are zero — check gradients!")

    # U_j = Jᵀ v_j / sqrt(λ_j) (normalised FIM eigenvector in param space)
    U_list = []
    for j in range(k_act):
        lam = L[j]
        vj  = torch.tensor(V[:, j], dtype=torch.float32)   # [n]
        uj  = J.T @ vj / (lam ** 0.5)                      # [d]
        uj  = uj / (uj.norm() + 1e-12)
        U_list.append(uj)
    U = torch.stack(U_list, dim=1)         # [d, k_act]
    return U, L[:k_act]


def compute_base_outputs(model, loader, normalizer, n_test):
    all_x, all_y = [], []
    n = 0
    with torch.no_grad():
        for x, y in loader:
            if n >= n_test:
                break
            x, _ = normalizer.preprocess(x, y)
            rem  = min(n_test - n, x.shape[0])
            x    = x[:rem]
            yh   = model(x.to(DEVICE)).cpu()
            all_x.append(x.cpu())
            all_y.append(yh)
            n += rem
    return torch.cat(all_x, 0), torch.cat(all_y, 0)


def perturb_and_measure(model, param_name, u_j, x_test, y_base):
    """θ* + ε·u_j → Δy → freq-band energies."""
    for pname, p in model.named_parameters():
        if pname != param_name:
            continue
        if p.is_complex():
            d = p.numel()
            re = u_j[:d].reshape(p.shape).to(device=p.device)
            im = u_j[d:].reshape(p.shape).to(device=p.device)
            delta = torch.complex(re, im).to(dtype=p.dtype)
        else:
            delta = u_j.reshape(p.shape).to(device=p.device, dtype=p.dtype)

        with torch.no_grad():
            p.data.add_(EPSILON * delta)

        y_pert_list = []
        with torch.no_grad():
            for st in range(0, x_test.shape[0], FWD_BATCH):
                xb = x_test[st:st+FWD_BATCH].to(DEVICE)
                y_pert_list.append(model(xb).cpu())
        y_pert = torch.cat(y_pert_list, 0)

        with torch.no_grad():
            p.data.sub_(EPSILON * delta)

        delta_y = y_pert - y_base
        return freq_band_energy(delta_y)

    raise ValueError(f"Parameter {param_name} not found in model")


def analyse_one_param(model, param_name, loader_fim, loader_test, normalizer):
    """
    Returns: low_fracs [K], eigenvalues [K], rho (float), pval (float)
    """
    log.info(f"\n  ── {param_name} ──")

    # ① FIM eigenvectors
    grads = collect_fim_gradients(model, loader_fim, normalizer, param_name, N_FIM)
    U, eigvals = top_k_eigenvectors(grads, K_EIGEN)
    k_act = U.shape[1]
    log.info(f"    λ_1={eigvals[0]:.3e}  λ_{k_act}={eigvals[-1]:.3e}")

    # ② base outputs
    x_test, y_base = compute_base_outputs(model, loader_test, normalizer, N_TEST)

    # ③ sweep eigenvectors
    low_fracs = []
    log.info(f"    Sweeping {k_act} eigenvectors …")
    for j in tqdm(range(k_act), desc=f"    [{param_name.split('.')[-2:]}]", leave=False):
        u_j = U[:, j].to(DEVICE)
        el, em, eh = perturb_and_measure(model, param_name, u_j, x_test, y_base)
        low_fracs.append(low_frac(el, em, eh))

    low_fracs = np.array(low_fracs)
    rho, pval = spearmanr(np.arange(k_act), low_fracs)
    supported = "✓" if rho < 0 else "✗"
    log.info(f"    f_low @j=1: {low_fracs[0]:.3f}  @j={k_act}: {low_fracs[-1]:.3f}")
    log.info(f"    Spearman ρ = {rho:+.3f}  (p={pval:.3e})  {supported}")

    return low_fracs, eigvals, rho, pval


# ═══════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════

LAYER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
LAYER_LABELS = [f"Layer {i+1}" for i in range(4)]

def make_figure(results):
    """
    results: list of (param_name, low_fracs, rho, pval) tuples.
    One clean panel: f_low vs FIM rank j for each spectral conv layer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    k = len(results[0][1])
    ranks = np.arange(1, k + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    all_lf = np.stack([r[1] for r in results], axis=0)   # [4, K]

    for i, (pname, lf, rho, pval) in enumerate(results):
        ax.plot(ranks, lf,
                color=LAYER_COLORS[i], lw=1.8, marker='o', markersize=4,
                label=f"{LAYER_LABELS[i]}  (ρ={rho:+.2f})")

    # Grand mean ± std
    mean_lf = all_lf.mean(axis=0)
    std_lf  = all_lf.std(axis=0)
    ax.plot(ranks, mean_lf, color="black", lw=2.5, ls="--", label="Mean", zorder=5)
    ax.fill_between(ranks, mean_lf - std_lf, mean_lf + std_lf,
                    color="black", alpha=0.10, zorder=4)

    # Overall trend
    overall_lf = all_lf.mean(axis=0)
    z = np.polyfit(ranks, overall_lf, 1)
    ax.plot(ranks, np.polyval(z, ranks), color="gray", lw=1.2, ls=":",
            zorder=3, label=None)

    # Overall Spearman ρ
    rho_all, pval_all = spearmanr(ranks, overall_lf)
    ax.text(0.97, 0.95,
            f"Overall ρ = {rho_all:+.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8))

    # Horizontal guide at 0.5
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax.set_xlabel("FIM eigenvector rank $j$", fontsize=12)
    ax.set_ylabel(r"Low-freq energy fraction $f_{\mathrm{low}}^{(j)}$", fontsize=12)
    ax.set_title("FIM principal subspace captures low-frequency physics\n"
                 "(FNO, Cylinder dataset)",
                 fontsize=11, pad=8)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(OUTPUT_DIR, f"rq1_fno_fim_theory.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("FNO FIM Low-Frequency Alignment  (PhysGuard theory)")
    log.info(f"  N_FIM={N_FIM}  N_TEST={N_TEST}  K_EIGEN={K_EIGEN}  ε={EPSILON}")
    log.info("=" * 60)

    # ── data ──────────────────────────────────────────────────────
    log.info("Loading Cylinder dataset …")
    train_ds, val_ds, normalizer = build_datasets()
    log.info(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    loader_fim  = DataLoader(train_ds, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    loader_test = DataLoader(val_ds,   batch_size=FWD_BATCH, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────
    log.info("Building FNO …")
    model = build_fno(train_ds)
    load_checkpoint(model, CKPT_PATH)
    model.to(DEVICE)
    model.eval()

    # Quick sanity: list target params
    for pname, p in model.named_parameters():
        if pname in TARGET_PARAMS:
            log.info(f"  {pname}: shape={tuple(p.shape)}  "
                     f"complex={p.is_complex()}  numel={p.numel()}")

    # ── per-layer analysis ─────────────────────────────────────────
    results = []   # (param_name, low_fracs, rho, pval)
    for pname in TARGET_PARAMS:
        lf, ev, rho, pval = analyse_one_param(
            model, pname, loader_fim, loader_test, normalizer)
        results.append((pname, lf, rho, pval))

    # ── summary ───────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info(f"{'Layer':<35}  {'ρ':>7}  {'p':>10}  {'supported'}")
    log.info("-" * 60)
    rhos = []
    for pname, lf, rho, pval in results:
        sup = "✓ YES" if rho < 0 else "✗  NO"
        log.info(f"  {pname:<33}  {rho:+.3f}  {pval:.3e}  {sup}")
        rhos.append(rho)
    mean_rho = np.mean(rhos)
    lfps     = -mean_rho
    log.info(f"\n  Mean Spearman ρ = {mean_rho:+.3f}")
    log.info(f"  LFPS            = {lfps:+.3f}  "
             f"({'positive → hypothesis SUPPORTED' if lfps > 0 else 'negative → NOT supported'})")
    log.info("=" * 60)

    # ── save ──────────────────────────────────────────────────────
    npz_path = os.path.join(OUTPUT_DIR, "rq1_fno_fim_theory_data.npz")
    save_dict = {}
    for i, (pname, lf, ev, rho) in enumerate(
            zip(TARGET_PARAMS,
                [r[1] for r in results],
                [r[2] for r in results],
                [r[3] for r in results])):
        safe = f"layer{i}"
        save_dict[f"{safe}_low_fracs"]  = lf
        save_dict[f"{safe}_rho"]        = np.array(rho)
    save_dict["lfps"]     = np.array(lfps)
    save_dict["mean_rho"] = np.array(mean_rho)
    np.savez(npz_path, **save_dict)
    log.info(f"Data saved → {npz_path}")

    # ── figure ────────────────────────────────────────────────────
    make_figure(results)
    log.info("Done.")


if __name__ == "__main__":
    main()
