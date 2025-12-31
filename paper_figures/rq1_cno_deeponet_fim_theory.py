#!/usr/bin/env python3
"""
CNO & DeepONet FIM Low-Frequency Alignment — Theory Figure for PhysGuard
=========================================================================
Same analysis as rq1_fno_fim_theory.py but for CNO and DeepONet.

CNO:     auto-selects top-4 conv layers by numel
DeepONet: explicitly uses BranchNet conv1~conv4 (clear depth progression)

Output:
  figures/rq1_cno_fim_theory.pdf / .png
  figures/rq1_deeponet_fim_theory.pdf / .png

Run:
  cd RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python \
      -u ../scripts/rq1_cno_deeponet_fim_theory.py
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
CKPT_BASE    = ("./results/001-cylinder")
OUTPUT_DIR   = "./figures"

CKPT_CNO      = f"{CKPT_BASE}/cno/cno_cylinder_pretrained/2026-03-11_09-49-04/model_5000.pth"
CKPT_DEEPONET = f"{CKPT_BASE}/deeponet/deeponet_cylinder_pretrained/2026-03-12_11-16-03/model_5000.pth"

# ─── hyper-params ───────────────────────────────────────────────────
N_FIM     = 50
N_TEST    = 50
K_EIGEN   = 20
EPSILON   = 1e-3
FWD_BATCH = 8

# DeepONet: all 4 BranchNet conv layers (depth progression: 32→64→128→256 channels)
DEEPONET_PARAMS = [
    "branch.conv1.0.weight",
    "branch.conv2.0.weight",
    "branch.conv3.0.weight",
    "branch.conv4.0.weight",
]
DEEPONET_LABELS = ["Conv1 (32ch)", "Conv2 (64ch)", "Conv3 (128ch)", "Conv4 (256ch)"]

# CNO: auto-selected top-4 by numel (set at runtime after model is loaded)
CNO_N_PARAMS = 4
CNO_MIN_NUMEL = 10_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  FFT UTILITIES  (identical to rq1_fno_fim_theory.py)
# ═══════════════════════════════════════════════════════════════════

_radial_cache = {}

def _build_radial_grid(T, H, W, device):
    half = min(T // 2, H // 2, W // 2)
    ii  = torch.arange(T // 2, device=device).float()
    jj  = torch.arange(H // 2, device=device).float()
    kk  = torch.arange(W // 2, device=device).float()
    gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing='ij')
    radial = torch.floor(torch.sqrt(gi**2 + gj**2 + gk**2)).long()
    valid  = radial < half
    return radial, valid, half

def freq_band_energy(delta_y: torch.Tensor):
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

def build_model(train_ds, **kwargs):
    from realpdebench.model.load_model import load_model
    return load_model(train_ds, device=DEVICE, **kwargs)

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

def select_cno_params(model, n_max=CNO_N_PARAMS, min_numel=CNO_MIN_NUMEL):
    """Auto-select top-n_max conv weight layers by numel, excluding bias/bn."""
    candidates = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.dim() < 2:
            continue
        low = name.lower()
        if any(t in low for t in ("bias", "norm", ".bn", "batch_norm")):
            continue
        n = p.numel()
        if n < min_numel:
            continue
        candidates.append((name, n))
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in candidates[:n_max]]
    return selected


# ═══════════════════════════════════════════════════════════════════
#  FIM ANALYSIS CORE
# ═══════════════════════════════════════════════════════════════════

def collect_fim_gradients(model, loader, normalizer, param_name, n_samples):
    grads = []
    n = 0
    short = param_name.split(".")[-2:]
    pbar  = tqdm(total=n_samples, desc=f"    FIM [{short}]", leave=False)
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
            gv = g.reshape(-1).float().cpu()
            grads.append(gv)
            break
        n += 1
        pbar.update(1)
    pbar.close()
    return grads

def top_k_eigenvectors(grads, k):
    J  = torch.stack(grads, dim=0)
    G  = J @ J.T
    G_np = G.numpy().astype(np.float64)
    L, V = np.linalg.eigh(G_np)
    L = L[::-1]; V = V[:, ::-1]
    L = np.maximum(L, 0)
    k_act = min(k, int((L > 1e-12).sum()))
    if k_act == 0:
        raise RuntimeError("All FIM eigenvalues are zero — check gradients!")
    U_list = []
    for j in range(k_act):
        lam = L[j]
        vj  = torch.tensor(V[:, j], dtype=torch.float32)
        uj  = J.T @ vj / (lam ** 0.5)
        uj  = uj / (uj.norm() + 1e-12)
        U_list.append(uj)
    return torch.stack(U_list, dim=1), L[:k_act]

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
            all_x.append(x.cpu()); all_y.append(yh)
            n += rem
    return torch.cat(all_x, 0), torch.cat(all_y, 0)

def perturb_and_measure(model, param_name, u_j, x_test, y_base):
    for pname, p in model.named_parameters():
        if pname != param_name:
            continue
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
    raise ValueError(f"Param {param_name} not found")

def analyse_one_param(model, param_name, loader_fim, loader_test, normalizer, label=None):
    tag = label or param_name
    log.info(f"\n  ── {tag} ──")
    grads = collect_fim_gradients(model, loader_fim, normalizer, param_name, N_FIM)
    U, eigvals = top_k_eigenvectors(grads, K_EIGEN)
    k_act = U.shape[1]
    log.info(f"    λ_1={eigvals[0]:.3e}  λ_{k_act}={eigvals[-1]:.3e}")

    x_test, y_base = compute_base_outputs(model, loader_test, normalizer, N_TEST)

    low_fracs = []
    for j in tqdm(range(k_act), desc=f"    sweep [{tag.split('.')[-1]}]", leave=False):
        u_j = U[:, j].to(DEVICE)
        el, em, eh = perturb_and_measure(model, param_name, u_j, x_test, y_base)
        low_fracs.append(low_frac(el, em, eh))

    low_fracs = np.array(low_fracs)
    rho, pval = spearmanr(np.arange(k_act), low_fracs)
    sup = "✓" if rho < 0 else "✗"
    log.info(f"    f_low @j=1: {low_fracs[0]:.3f}  @j={k_act}: {low_fracs[-1]:.3f}")
    log.info(f"    Spearman ρ = {rho:+.3f}  (p={pval:.3e})  {sup}")
    return low_fracs, eigvals, rho, pval


# ═══════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def make_figure(results, model_name, out_base):
    """
    results: list of (label, low_fracs, rho, pval)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k     = len(results[0][1])
    ranks = np.arange(1, k + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    all_lf = np.stack([r[1] for r in results], axis=0)

    for i, (label, lf, rho, pval) in enumerate(results):
        ax.plot(ranks, lf,
                color=COLORS[i], lw=1.8, marker='o', markersize=4,
                label=f"{label}  (ρ={rho:+.2f})")

    mean_lf = all_lf.mean(axis=0)
    std_lf  = all_lf.std(axis=0)
    ax.plot(ranks, mean_lf, color="black", lw=2.5, ls="--", label="Mean", zorder=5)
    ax.fill_between(ranks, mean_lf - std_lf, mean_lf + std_lf,
                    color="black", alpha=0.10, zorder=4)

    overall_lf = all_lf.mean(axis=0)
    rho_all, _ = spearmanr(ranks, overall_lf)
    ax.text(0.97, 0.97,
            f"Overall ρ = {rho_all:+.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8))

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax.set_xlabel("FIM eigenvector rank $j$", fontsize=12)
    ax.set_ylabel(r"Low-freq energy fraction $f_{\mathrm{low}}^{(j)}$", fontsize=12)
    ax.set_title(f"FIM principal subspace captures low-frequency physics\n"
                 f"({model_name}, Cylinder dataset)", fontsize=11, pad=8)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def run_model(arch_name, model, ckpt_path, target_params, param_labels,
              loader_fim, loader_test, normalizer):
    log.info(f"\n{'='*60}")
    log.info(f"  {arch_name}")
    log.info(f"{'='*60}")

    load_checkpoint(model, ckpt_path)
    model.to(DEVICE)
    model.eval()

    # Log target param shapes
    param_dict = dict(model.named_parameters())
    for pname in target_params:
        if pname in param_dict:
            p = param_dict[pname]
            log.info(f"  {pname}: {tuple(p.shape)}  numel={p.numel()}")
        else:
            log.warning(f"  {pname}: NOT FOUND — skipping")

    results = []
    for pname, label in zip(target_params, param_labels):
        if pname not in param_dict:
            continue
        lf, ev, rho, pval = analyse_one_param(
            model, pname, loader_fim, loader_test, normalizer, label=label)
        results.append((label, lf, rho, pval))

    # Summary
    log.info(f"\n{'─'*60}")
    log.info(f"  {arch_name} SUMMARY")
    rhos = []
    for label, lf, rho, pval in results:
        sup = "✓ YES" if rho < 0 else "✗  NO"
        log.info(f"    {label:<25}  ρ={rho:+.3f}  p={pval:.3e}  {sup}")
        rhos.append(rho)
    lfps = -np.mean(rhos)
    log.info(f"\n  LFPS = {lfps:+.3f}")

    return results


def main():
    log.info("=" * 60)
    log.info("CNO & DeepONet FIM Low-Frequency Alignment")
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
    cno_model = build_model(train_ds, model_name="cno", N_layers=3)
    # Find target params after model is built
    cno_params = select_cno_params(cno_model)
    # Create short readable labels
    cno_labels = [f"{p.split('.')[-3] if len(p.split('.')) >= 3 else p.split('.')[-2]}.{p.split('.')[-1]}"
                  for p in cno_params]
    # Truncate label to be readable
    cno_labels = [lbl[:22] for lbl in cno_labels]

    log.info(f"\nCNO auto-selected params ({len(cno_params)}):")
    for p, lbl in zip(cno_params, cno_labels):
        log.info(f"  [{lbl}] ← {p}")

    cno_results = run_model(
        "CNO", cno_model, CKPT_CNO,
        cno_params, cno_labels,
        loader_fim, loader_test, normalizer)

    del cno_model
    torch.cuda.empty_cache()

    make_figure(cno_results, "CNO",
                os.path.join(OUTPUT_DIR, "rq1_cno_fim_theory"))

    # ─── DeepONet ─────────────────────────────────────────────────
    deeponet_model = build_model(train_ds, model_name="deeponet", p=128, dropout_rate=0.1)

    deeponet_results = run_model(
        "DeepONet", deeponet_model, CKPT_DEEPONET,
        DEEPONET_PARAMS, DEEPONET_LABELS,
        loader_fim, loader_test, normalizer)

    del deeponet_model
    torch.cuda.empty_cache()

    make_figure(deeponet_results, "DeepONet",
                os.path.join(OUTPUT_DIR, "rq1_deeponet_fim_theory"))

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
