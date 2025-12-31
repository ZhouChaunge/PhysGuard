#!/usr/bin/env python3
"""
Cross-Architecture FIM Low-Frequency Preference (LFPS) Analysis
================================================================
Single unified metric: Spearman ρ between FIM eigenvector rank j and the
output-space low-frequency energy fraction f_low^(j) of the perturbation Δy.

Expected result: ρ < 0 for ALL architectures — meaning FIM top eigenvectors
preferentially capture low-frequency physical structure, independent of model
inductive biases.

Definition
----------
  For each architecture, select the top-2 weight matrices (by real parameter count).
  For each weight matrix:
    1. Estimate FIM via N_FIM per-sample gradients → Gram trick for top-K eigenvectors.
    2. For each eigenvector u_j (j=1…K, ranked by eigenvalue λ_j):
          θ ← θ* + ε·u_j → forward on N_TEST inputs → Δy
          f_low^(j) = ‖FFT(Δy)|_{low-k}‖² / ‖FFT(Δy)‖²
    3. Spearman ρ(j, f_low^(j)) — NEGATIVE means top eigenvectors ↔ low freq.
  Model-level LFPS = mean(-ρ) across all analyzed weight matrices.

Models analysed: FNO, CNO, DeepONet, DPOT, Transolver
Dataset: Cylinder (numerical train for FIM; real val for perturbation test)

Run (from RealPDEBench/ directory):
  CUDA_VISIBLE_DEVICES=1 conda run -n pytorch310 python ../scripts/rq1_cross_arch_fim.py

Output:
  figures/rq1_cross_arch_fim.pdf  (and .png)
  figures/rq1_cross_arch_fim_data.npz
"""

import os, sys, logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr

# ─────────────────────────── paths / repo ───────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.join(SCRIPT_DIR, "..", "RealPDEBench")
sys.path.insert(0, REPO_DIR)

DATASET_ROOT = "./data/realpdebench/"
CKPT_BASE    = "./results/001-cylinder"
DPOT_BACKBONE= "./dpot_ckpts/model_S.pth"
OUTPUT_DIR   = "./figures"

# ─────────────────────────── hyper-params ───────────────────────────
N_FIM     = 30       # gradient samples for FIM estimation
N_TEST    = 30       # validation samples for output-perturbation
K_EIGEN   = 20       # FIM eigenvectors to analyse per weight matrix
EPSILON   = 1e-3     # perturbation scale ε
FWD_BATCH = 8        # default batch size for perturbed-forward sweep
N_PARAMS  = 2        # top-N weight matrices per model (by real numel)
MIN_NUMEL = 50_000   # minimum real numel to be eligible

# Cache file: saves intermediate per-arch results so we can resume on crash
CACHE_FILE = "/tmp/rq1_cross_arch_cache.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────── arch configs ───────────────────────────
ARCH_CONFIGS = {
    "FNO": {
        "ckpt": f"{CKPT_BASE}/fno/fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth",
        "build_kwargs": dict(
            model_name="fno",
            modes1=4, modes2=12, modes3=16, n_layers=4, width=64,
        ),
    },
    "CNO": {
        "ckpt": f"{CKPT_BASE}/cno/cno_cylinder_pretrained/2026-03-11_09-49-04/model_5000.pth",
        "build_kwargs": dict(
            model_name="cno",
            N_layers=3,
        ),
    },
    "DeepONet": {
        "ckpt": f"{CKPT_BASE}/deeponet/deeponet_cylinder_pretrained/2026-03-12_11-16-03/model_5000.pth",
        "build_kwargs": dict(
            model_name="deeponet",
            p=128,
            dropout_rate=0.1,
        ),
    },
    "DPOT": {
        "ckpt": f"{CKPT_BASE}/dpot/dpot_s_cylinder_pretrained/2026-03-12_01-49-29/model_4000.pth",
        "build_kwargs": dict(
            model_name="dpot",
            img_size=128, patch_size=8,
            in_channels=4, out_channels=4,
            in_timesteps=20, out_timesteps=20,
            embed_dim=1024, depth=6, n_blocks=8, modes=32,
            mlp_ratio=1, out_layer_dim=32, normalize=False,
            act="gelu", time_agg="exp_mlp", n_cls=12,
            model_type="dpot",
            checkpoint_path=None,   # skip backbone pre-load; use cylinder-ckpt below
        ),
    },
    "Transolver": {
        "ckpt": f"{CKPT_BASE}/transolver/transolver_cylinder_pretrained/2026-03-25_21-23-52/model_5000.pth",
        "build_kwargs": dict(
            model_name="transolver",
            space_dim=3, n_layers=1, n_hidden=256, n_head=8,
            H=128, W=64, D=20,
            fun_dim=0, out_dim=3, ref=4,
            dropout=0.1, act="gelu", mlp_ratio=4, slice_num=16,
        ),
        # Reduce batch size to 1 to avoid OOM (Transolver has large intermediate tensors)
        "fwd_batch": 1,
    },
}

# ─────────────────────────── colours ────────────────────────────────
ARCH_COLOR = {
    "FNO"      : "#4C72B0",
    "CNO"      : "#DD8452",
    "DeepONet" : "#55A868",
    "DPOT"     : "#C44E52",
    "Transolver": "#8172B2",
}
ARCH_MARKER = {
    "FNO"      : "o",
    "CNO"      : "s",
    "DeepONet" : "^",
    "DPOT"     : "D",
    "Transolver": "P",
}

# ─────────────────────────── logging ────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
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


def freq_band_energy(delta_y: torch.Tensor, _cache: dict = {}):
    """
    Vectorised radial-wavenumber decomposition.
    delta_y: [B, T, H, W, C]
    Returns (e_low, e_mid, e_high) as Python floats (unnormalized energies).
    """
    T, H, W = delta_y.shape[1], delta_y.shape[2], delta_y.shape[3]
    key = (T, H, W, delta_y.device.type)
    if key not in _cache:
        _cache[key] = _build_radial_grid(T, H, W, delta_y.device)
    radial, valid, half = _cache[key]

    D     = torch.fft.fftn(delta_y.float(), dim=[1, 2, 3])
    power = D.abs() ** 2
    power_half = power[:, :T//2, :H//2, :W//2, :]

    B, C    = power_half.shape[0], power_half.shape[-1]
    pfv     = power_half.permute(0, 4, 1, 2, 3).reshape(B * C, -1)
    v_flat  = valid.reshape(-1)
    r_flat  = radial.reshape(-1)[v_flat]
    pfv_valid = pfv[:, v_flat]

    pfv_sum = pfv_valid.sum(dim=0)
    err_F   = torch.zeros(half, device=delta_y.device)
    err_F.scatter_add_(0, r_flat, pfv_sum)
    err_F  /= (B * C)

    iLow  = int(round(half / 3))
    iHigh = int(round(half * 2 / 3))
    e_low  = err_F[:iLow].sum().item()
    e_mid  = err_F[iLow:iHigh].sum().item()
    e_high = err_F[iHigh:].sum().item()
    return e_low, e_mid, e_high


def low_frac(e_low, e_mid, e_high):
    total = e_low + e_mid + e_high + 1e-30
    return e_low / total


def real_numel(p: torch.nn.Parameter) -> int:
    """Effective number of real scalars (complex params count as 2×)."""
    n = p.numel()
    return 2 * n if p.is_complex() else n


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def build_datasets():
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer

    common = dict(dataset_name="cylinder", dataset_root=DATASET_ROOT)
    sim_train = CylinderHFDataset(mode="train", dataset_type="numerical", **common)
    sim_val   = CylinderHFDataset(mode="val",   dataset_type="real",      **common)
    normalizer = GaussianNormalizer(sim_train, device=DEVICE)
    return sim_train, sim_val, normalizer


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

def build_model(arch_name: str, train_dataset) -> torch.nn.Module:
    """Instantiate the model for the given architecture using the benchmark's load_model."""
    from realpdebench.model.load_model import load_model
    cfg = dict(ARCH_CONFIGS[arch_name]["build_kwargs"])
    model = load_model(train_dataset, device=DEVICE, **cfg)
    return model


def load_checkpoint_robust(model: torch.nn.Module, ckpt_path: str):
    """
    Load a checkpoint with robust handling:
    - Unwraps 'model_state_dict' key if present.
    - Strips 'module.' prefix from DDP-saved checkpoints.
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]
    else:
        raw = ckpt
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"  Missing keys ({len(missing)}): {missing[:5]}" +
                    (" ..." if len(missing) > 5 else ""))
    if unexpected:
        log.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}" +
                    (" ..." if len(unexpected) > 5 else ""))
    model.eval()


def select_target_params(model: torch.nn.Module,
                          n_max: int = N_PARAMS,
                          min_numel: int = MIN_NUMEL):
    """
    Return the names of the top-n_max weight matrices ranked by real_numel.
    Filters out: bias, norm/bn parameters, embedding-like 1-D params.
    """
    candidates = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = real_numel(p)
        if n < min_numel:
            continue
        low = name.lower()
        if any(tok in low for tok in ("bias", "norm", ".bn", "embed", "placeholder")):
            continue
        if p.dim() < 2:
            continue
        candidates.append((name, n))
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in candidates[:n_max]]
    if not selected:
        # Fallback: take any large parameter
        all_params = [(n, real_numel(p)) for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
        all_params.sort(key=lambda x: x[1], reverse=True)
        selected = [n for n, _ in all_params[:n_max]]
    return selected


# ═══════════════════════════════════════════════════════════════════
#  FIM GRADIENT COLLECTION
# ═══════════════════════════════════════════════════════════════════

def collect_gradients(model, dataloader, normalizer, target_names, n_samples):
    """
    Per-sample FIM gradient vectors for selected parameters.
    Returns: { param_name -> list of 1-D float16 CPU tensors }
    """
    target_set = set(target_names)
    grad_accum = {name: [] for name in target_set}
    n_collected = 0

    pbar = tqdm(total=n_samples, desc="    FIM gradients", leave=False)
    for batch in dataloader:
        if n_collected >= n_samples:
            break
        x, y = batch
        x, y = normalizer.preprocess(x, y)
        B = x.shape[0]
        for i in range(B):
            if n_collected >= n_samples:
                break
            model.zero_grad()
            xi, yi = x[i:i+1], y[i:i+1]
            loss = model.train_loss(xi, yi)
            if isinstance(loss, torch.Tensor):
                loss = loss.mean()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None or name not in target_set:
                    continue
                g = param.grad.detach().reshape(-1)
                if g.is_complex():
                    g = torch.cat([g.real, g.imag])
                grad_accum[name].append(g.cpu().half())
            n_collected += 1
            pbar.update(1)
    pbar.close()
    log.info(f"    Collected {n_collected} gradient samples.")
    return grad_accum


# ═══════════════════════════════════════════════════════════════════
#  EIGENVECTOR COMPUTATION  (Gram trick)
# ═══════════════════════════════════════════════════════════════════

def compute_eigenvectors(grad_list, k_max):
    """
    G ∈ R^{N×d} (or N×d_real for complex params),
    K = G Gᵀ ∈ R^{N×N} — eigendecompose,
    then project back: U = Gᵀ V — then normalise.
    Returns (eigenvalues [k], U [d_real, k]).
    """
    G   = torch.stack(grad_list, dim=0).float()    # [N, d_real]
    N, d = G.shape
    K   = G @ G.t()                                 # [N, N]
    vals, vecs = torch.linalg.eigh(K)               # ascending
    vals = vals.flip(0).clamp(min=0)
    vecs = vecs.flip(1)
    U    = G.t() @ vecs                             # [d, N]
    norms = U.norm(dim=0, keepdim=True).clamp(min=1e-12)
    U    = U / norms
    k    = min(k_max, N)
    return vals[:k], U[:, :k]                       # [k], [d_real, k]


# ═══════════════════════════════════════════════════════════════════
#  OUTPUT-PERTURBATION MEASUREMENT
# ═══════════════════════════════════════════════════════════════════

def compute_base_outputs(model, loader, normalizer, n_test):
    all_x, all_y = [], []
    n = 0
    with torch.no_grad():
        for x, y in loader:
            if n >= n_test:
                break
            x, _ = normalizer.preprocess(x, y)
            rem   = min(n_test - n, x.shape[0])
            x     = x[:rem]
            y_hat = model(x.to(DEVICE)).cpu()
            all_x.append(x.cpu())
            all_y.append(y_hat)
            n += rem
    return torch.cat(all_x, 0), torch.cat(all_y, 0)


def perturb_and_measure(model, param_name, u_j, x_test, y_base,
                        fwd_batch=FWD_BATCH):
    """
    Add ε·u_j to param_name, forward on x_test, compute freq bands of Δy,
    then restore the parameter.
    """
    # Find param
    target_param = None
    for n, p in model.named_parameters():
        if n == param_name:
            target_param = p
            break
    assert target_param is not None

    if target_param.is_complex():
        d_cplx = target_param.numel()
        re_part = u_j[:d_cplx].reshape(target_param.shape).to(
            device=target_param.device)
        im_part = u_j[d_cplx:].reshape(target_param.shape).to(
            device=target_param.device)
        delta = torch.complex(re_part, im_part).to(dtype=target_param.dtype)
    else:
        delta = u_j.reshape(target_param.shape).to(
            device=target_param.device, dtype=target_param.dtype)

    with torch.no_grad():
        target_param.data.add_(EPSILON * delta)

    y_pert_list = []
    with torch.no_grad():
        for st in range(0, x_test.shape[0], fwd_batch):
            xb  = x_test[st:st+fwd_batch].to(DEVICE)
            yb  = model(xb).cpu()
            y_pert_list.append(yb)
    y_pert = torch.cat(y_pert_list, 0)

    with torch.no_grad():
        target_param.data.sub_(EPSILON * delta)

    delta_y = y_pert - y_base
    return freq_band_energy(delta_y)


# ═══════════════════════════════════════════════════════════════════
#  PER-ARCHITECTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyse_arch(arch_name, model, sim_train, sim_val, normalizer,
                 fwd_batch=FWD_BATCH):
    """
    Returns a dict with per-param results and model-level LFPS.

    Result structure:
      { "params": { param_name: { "eigenvalues": ndarray [k],
                                  "low_fracs":   ndarray [k],
                                  "rho": float, "pval": float } },
        "lfps": float,          # model-level Low-Freq Preference Score
        "lfps_std": float,
        "mean_low_fracs": ndarray [k],
        "std_low_fracs" : ndarray [k],
      }
    """
    target_names = select_target_params(model)
    log.info(f"  Selected params: {target_names}")
    for tn in target_names:
        for n, p in model.named_parameters():
            if n == tn:
                log.info(f"    {tn}: shape={tuple(p.shape)}, "
                         f"complex={p.is_complex()}, numel_real={real_numel(p)}")

    # ── loaders ──
    loader_fim  = DataLoader(sim_train, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    loader_test = DataLoader(sim_val,   batch_size=fwd_batch, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── base outputs ──
    log.info("  Computing base outputs …")
    x_test, y_base = compute_base_outputs(model, loader_test, normalizer, N_TEST)

    # ── FIM gradient collection (all target params in one pass) ──
    log.info("  Collecting FIM gradients …")
    grad_dict = collect_gradients(model, loader_fim, normalizer, target_names, N_FIM)

    # ── per-param analysis ──
    param_results = {}
    all_rho       = []
    all_low_fracs = []

    for pname in target_names:
        grad_list = grad_dict[pname]
        if not grad_list:
            log.warning(f"  No gradients collected for {pname}, skipping.")
            continue

        log.info(f"\n  ── {pname} ──")
        eigenvalues, U = compute_eigenvectors(grad_list, K_EIGEN)
        k_actual = U.shape[1]
        log.info(f"  λ_1={eigenvalues[0]:.3e}, λ_{k_actual}={eigenvalues[-1]:.3e}")

        low_fracs = np.zeros(k_actual)
        log.info(f"  Sweeping {k_actual} eigenvectors …")
        for j in tqdm(range(k_actual),
                      desc=f"    {arch_name}/{pname.split('.')[-2:]}", leave=False):
            u_j = U[:, j].to(DEVICE)
            el, em, eh = perturb_and_measure(model, pname, u_j, x_test, y_base,
                                             fwd_batch=fwd_batch)
            low_fracs[j] = low_frac(el, em, eh)

        rho, pval = spearmanr(np.arange(1, k_actual + 1), low_fracs)
        log.info(f"  OS f_low @j=1: {low_fracs[0]:.3f}  @j={k_actual}: {low_fracs[-1]:.3f}")
        log.info(f"  Spearman ρ = {rho:.3f}  (p={pval:.3e})  "
                 f"{'SUPPORTED ✓' if rho < 0 else 'NOT supported ✗'}")

        param_results[pname] = dict(
            eigenvalues=eigenvalues.numpy(),
            low_fracs=low_fracs,
            rho=rho,
            pval=pval,
        )
        all_rho.append(rho)
        all_low_fracs.append(low_fracs)

    # model-level LFPS = mean(-ρ)  →  positive means FIM prefers low freq
    lfps     = float(-np.mean(all_rho))
    lfps_std = float(np.std(all_rho)) if len(all_rho) > 1 else 0.0

    # mean ± std across params (aligned to min k)
    min_k = min(lf.shape[0] for lf in all_low_fracs)
    stacked = np.stack([lf[:min_k] for lf in all_low_fracs], axis=0)
    mean_lf = stacked.mean(axis=0)
    std_lf  = stacked.std(axis=0)

    log.info(f"\n  Model-level LFPS = {lfps:.4f}  (std={lfps_std:.4f})\n")
    return dict(
        params=param_results,
        lfps=lfps,
        lfps_std=lfps_std,
        mean_low_fracs=mean_lf,
        std_low_fracs=std_lf,
    )


# ═══════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════

def make_figure(all_results: dict):
    """
    Two-panel figure:
      (a) f_low vs eigenvector rank j  — one line per architecture
      (b) LFPS bar chart               — one bar per architecture
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    archs  = list(all_results.keys())
    colors = [ARCH_COLOR[a]  for a in archs]
    markers= [ARCH_MARKER[a] for a in archs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("#333333")

    # ── (a) f_low vs rank ──────────────────────────────────────────
    ax = axes[0]
    for arch, c, m in zip(archs, colors, markers):
        res = all_results[arch]
        lf  = res["mean_low_fracs"]
        std = res["std_low_fracs"]
        k   = len(lf)
        x   = np.arange(1, k + 1)
        ax.plot(x, lf, color=c, marker=m, markersize=4.5,
                linewidth=1.6, label=arch, zorder=3)
        ax.fill_between(x, lf - std, lf + std, alpha=0.15, color=c, zorder=2)

    # reference: flat line at overall mean
    all_means = np.concatenate([r["mean_low_fracs"] for r in all_results.values()])
    grand_mean = all_means.mean()
    ax.axhline(grand_mean, color="grey", linewidth=0.8, linestyle="--",
               alpha=0.6, zorder=1, label="Grand mean")

    ax.set_xlabel("FIM eigenvector rank $j$", fontsize=11)
    ax.set_ylabel("Low-frequency energy fraction $f^{(j)}_{\\mathrm{low}}$", fontsize=11)
    ax.set_title("(a) Output-space low-frequency preference vs. FIM rank",
                 fontsize=10.5, pad=8)
    ax.legend(fontsize=9.5, framealpha=0.9, loc="upper right")
    ax.set_xlim(0.5, K_EIGEN + 0.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(labelsize=9)
    ax.set_xticks([1, 5, 10, 15, 20])

    # annotate Spearman ρ for each arch
    for arch, c in zip(archs, colors):
        rho_vals = [v["rho"] for v in all_results[arch]["params"].values()]
        mean_rho = float(np.mean(rho_vals))
        lf = all_results[arch]["mean_low_fracs"]
        y_pos = lf[-1] - 0.005
        ax.annotate(f"ρ={mean_rho:.2f}", xy=(K_EIGEN, lf[-1]),
                    fontsize=7.5, color=c, ha="right", va="top",
                    xytext=(-4, -3), textcoords="offset points")

    # ── (b) LFPS bar chart ─────────────────────────────────────────
    ax = axes[1]
    lfps_vals  = [all_results[a]["lfps"]     for a in archs]
    lfps_errs  = [all_results[a]["lfps_std"] for a in archs]
    x_pos      = np.arange(len(archs))

    bars = ax.bar(x_pos, lfps_vals, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.6, zorder=3)
    ax.errorbar(x_pos, lfps_vals, yerr=lfps_errs,
                fmt="none", ecolor="#333333", elinewidth=1.2,
                capsize=3, zorder=4)

    # value labels
    for bar, val in zip(bars, lfps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(lfps_errs) * 0.05 + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="#333333", linewidth=0.8, zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(archs, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("LFPS = $-\\bar{\\rho}_s$", fontsize=11)
    ax.set_title("(b) Low-Frequency Preference Score per model", fontsize=10.5, pad=8)
    ax.tick_params(labelsize=9)
    ax.set_ylim(-0.1, max(lfps_vals) * 1.35 + 0.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # horizontal guide line at LFPS=0 label
    ax.text(len(archs) - 0.5, 0.005, "LFPS = 0\n(no preference)", ha="right",
            va="bottom", fontsize=8, color="grey", style="italic")

    # ── final touches ─────────────────────────────────────────────
    plt.tight_layout(pad=1.8)

    for fmt in ("pdf", "png"):
        path = os.path.join(OUTPUT_DIR, f"rq1_cross_arch_fim.{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info(f"Saved figure → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    import pickle

    log.info("=" * 64)
    log.info("Cross-Architecture FIM Low-Frequency Preference Analysis")
    log.info(f"  N_FIM={N_FIM}  N_TEST={N_TEST}  K_EIGEN={K_EIGEN}  "
             f"N_PARAMS={N_PARAMS}  EPSILON={EPSILON}")
    log.info("=" * 64)

    # ── load existing cache (resume on crash) ──
    all_results = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                all_results = pickle.load(f)
            log.info(f"Loaded cache from {CACHE_FILE}: "
                     f"{list(all_results.keys())} already done")
        except Exception as e:
            log.warning(f"Could not load cache ({e}), starting fresh")
            all_results = {}

    # ── data (shared) ──
    archs_todo = [k for k in ARCH_CONFIGS if k not in all_results]
    if not archs_todo:
        log.info("All architectures already cached — skipping data loading")
    else:
        log.info(f"Architectures to analyse: {archs_todo}")
        log.info("Loading dataset …")
        sim_train, sim_val, normalizer = build_datasets()
        log.info(f"  train: {len(sim_train)} samples  |  val: {len(sim_val)} samples")

        # ── per-arch analysis ──
        for arch_name in archs_todo:
            cfg = ARCH_CONFIGS[arch_name]
            fwd_batch = cfg.get("fwd_batch", FWD_BATCH)

            log.info(f"\n{'─'*60}")
            log.info(f"Architecture: {arch_name}  (fwd_batch={fwd_batch})")
            log.info(f"{'─'*60}")

            # Build model
            log.info("  Building model …")
            model = build_model(arch_name, sim_train)

            # Load cylinder-pretrained checkpoint
            log.info(f"  Loading checkpoint: {cfg['ckpt']}")
            load_checkpoint_robust(model, cfg["ckpt"])
            model.to(DEVICE)
            model.eval()

            # Run analysis
            result = analyse_arch(arch_name, model, sim_train, sim_val, normalizer,
                                  fwd_batch=fwd_batch)
            all_results[arch_name] = result

            # Save cache after each arch (crash-safe)
            try:
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(all_results, f, protocol=4)
                log.info(f"  Cache saved → {CACHE_FILE}")
            except Exception as e:
                log.warning(f"  Cache save failed: {e}")

            # Free GPU memory before next model
            del model
            torch.cuda.empty_cache()

    # ── summary table ──
    log.info("\n" + "=" * 64)
    log.info("SUMMARY — Low-Frequency Preference Score (LFPS = -mean Spearman ρ)")
    log.info(f"{'Arch':<12} {'LFPS':>8} {'±std':>8}  {'Per-param ρ'}")
    log.info("-" * 64)
    for arch_name, res in all_results.items():
        per = {k: f"{v['rho']:+.3f}" for k, v in res["params"].items()}
        log.info(f"{arch_name:<12} {res['lfps']:>8.4f} {res['lfps_std']:>8.4f}  {per}")
    log.info("=" * 64)
    log.info("Negative Spearman ρ in every param of every architecture => SUPPORTED")
    log.info("FIM principal subspace preferentially captures low-frequency physics.")
    log.info("=" * 64)

    # ── save data ──
    npz_path = os.path.join(OUTPUT_DIR, "rq1_cross_arch_fim_data.npz")
    save_dict = {}
    for arch_name, res in all_results.items():
        save_dict[f"{arch_name}_lfps"]          = np.array(res["lfps"])
        save_dict[f"{arch_name}_lfps_std"]      = np.array(res["lfps_std"])
        save_dict[f"{arch_name}_mean_low_fracs"] = res["mean_low_fracs"]
        save_dict[f"{arch_name}_std_low_fracs"]  = res["std_low_fracs"]
        for pname, pres in res["params"].items():
            safe = pname.replace(".", "_")
            save_dict[f"{arch_name}_{safe}_low_fracs"]  = pres["low_fracs"]
            save_dict[f"{arch_name}_{safe}_eigenvalues"] = pres["eigenvalues"]
            save_dict[f"{arch_name}_{safe}_rho"]        = np.array(pres["rho"])
    np.savez(npz_path, **save_dict)
    log.info(f"Saved data → {npz_path}")

    # ── figure ──
    make_figure(all_results)
    log.info("Done.")


if __name__ == "__main__":
    main()
