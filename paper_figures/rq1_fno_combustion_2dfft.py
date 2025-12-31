#!/usr/bin/env python3
"""
FNO / Combustion — f_low per FIM rank, 2D spatial FFT
=====================================================
Replicates the Ctrl-Cylinder 2D-FFT analysis for turbulent combustion,
using the same metric as Cylinder/Ctrl-Cylinder so all three datasets
can be compared on the same axis.

Output:
  figures/rq1_fno_combustion_2dfft_data.npz
    keys: f_low [K], rho (float), pval (float), half (int)

Run:
  cd ./RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python \
      -u ../scripts/rq1_fno_combustion_2dfft.py
"""

import os, sys, logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.join(SCRIPT_DIR, "..", "RealPDEBench")
sys.path.insert(0, REPO_DIR)

DATASET_ROOT = "./data/realpdebench/"
CKPT_PATH    = ("./results/003-combustion"
                "/fno/fno_combustion_pretrained/2026-03-20_11-54-28/model_2000.pth")
OUTPUT_DIR   = "./figures"
OUT_NPZ      = f"{OUTPUT_DIR}/rq1_fno_combustion_2dfft_data.npz"

TARGET_PARAM = "spectral_convs.1.weights1"  # consistent with Cylinder/Ctrl-Cylinder
FNO_KWARGS   = dict(model_name="fno", modes1=4, modes2=16, modes3=16,
                    n_layers=4, width=64)

N_FIM     = 80
K         = 20
N_TEST    = 30
EPSILON   = 1e-3
FWD_BATCH = 4
SEED      = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── 2D spatial FFT (time-averaged) ────────────────────────────────

_grid_cache = {}

def _build_grid(H, W, device):
    half = min(H // 2, W // 2)
    jj   = torch.arange(H // 2, device=device).float()
    kk   = torch.arange(W // 2, device=device).float()
    gj, gk = torch.meshgrid(jj, kk, indexing='ij')
    radial = torch.floor(torch.sqrt(gj**2 + gk**2)).long()
    valid  = radial < half
    return radial, valid, half


def measure_f_low(delta_y: torch.Tensor) -> float:
    """delta_y: [B, T, H, W, C]  →  f_low (2D spatial FFT, time-averaged)"""
    H, W = delta_y.shape[2], delta_y.shape[3]
    key  = (H, W, delta_y.device.type)
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


# ── Data & model ─────────────────────────────────────────────────

def build_all():
    from realpdebench.data.combustion_hf_dataset import CombustionHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer
    from realpdebench.model.load_model import load_model

    common   = dict(dataset_name="combustion", dataset_root=DATASET_ROOT)
    train_ds = CombustionHFDataset(mode="train", dataset_type="numerical", **common)
    val_ds   = CombustionHFDataset(mode="val",   dataset_type="real",      **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)

    model = load_model(train_ds, device=DEVICE, **FNO_KWARGS)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    raw   = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    log.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    sample_x, sample_y = train_ds[0]
    log.info(f"Data shape: x={sample_x.shape}  y={sample_y.shape}")
    return train_ds, val_ds, normalizer, model


# ── FIM eigenvectors ──────────────────────────────────────────────

def collect_grads(model, loader, normalizer, n):
    grads = []
    pbar  = tqdm(total=n, desc=f"  FIM grads [{TARGET_PARAM}]", leave=False)
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
            gv = (torch.cat([p.grad.real.reshape(-1), p.grad.imag.reshape(-1)])
                  if p.is_complex() else p.grad.reshape(-1))
            grads.append(gv.float().cpu())
            break
        cnt += 1; pbar.update(1)
    pbar.close()
    log.info(f"  {len(grads)} gradient vectors, dim={grads[0].shape[0]}")
    return grads


def compute_eigenvectors(grads, k):
    J  = torch.stack(grads, dim=0)          # [N, d]
    G  = (J @ J.T).numpy().astype(np.float64)
    L, V = np.linalg.eigh(G)
    L  = np.maximum(L[::-1], 0); V = V[:, ::-1]
    n_valid = int((L > 1e-12 * L[0]).sum())
    log.info(f"  Valid eigenvectors: {n_valid}")

    U_list = []
    for j in range(min(k, n_valid)):
        lam = L[j]
        if lam < 1e-30: continue
        vj = torch.tensor(V[:, j], dtype=torch.float32)
        uj = J.T @ vj / (lam ** 0.5)
        uj = uj / (uj.norm() + 1e-12)
        U_list.append(uj)
    U = torch.stack(U_list, dim=1)       # [d, k]
    log.info(f"  Subspace dim: {U.shape[1]}")
    return U, L


# ── Perturbation measurement ──────────────────────────────────────

def get_param_info(model):
    for name, p in model.named_parameters():
        if name == TARGET_PARAM:
            return name, p
    raise RuntimeError(f"Param {TARGET_PARAM} not found")


def measure_eigvec_f_low(model, U, val_loader, normalizer, n_test):
    """Returns f_low[j] for each FIM eigenvector j."""
    pname, param = get_param_info(model)
    d = U.shape[0]
    is_complex = param.is_complex()
    if is_complex:
        assert d == param.numel() * 2, f"dim mismatch: d={d}, numel={param.numel()}"

    # collect test inputs
    xs = []
    for x, y in val_loader:
        x, _ = normalizer.preprocess(x, y)
        xs.append(x.to(DEVICE))
        if sum(xx.shape[0] for xx in xs) >= n_test:
            break
    xs = torch.cat(xs, dim=0)[:n_test]

    with torch.no_grad():
        y_base = model(xs)

    f_lows = []
    K_act  = U.shape[1]
    for j in tqdm(range(K_act), desc="  Measuring f_low per rank"):
        uj = U[:, j].to(DEVICE)
        orig = param.data.clone()

        if is_complex:
            half_d = param.numel()
            delta_r = uj[:half_d].reshape(param.shape)
            delta_i = uj[half_d:].reshape(param.shape)
            delta   = torch.complex(delta_r, delta_i) * EPSILON
        else:
            delta = uj.reshape(param.shape) * EPSILON

        with torch.no_grad():
            param.data.add_(delta)
            y_pert = model(xs)
            param.data.copy_(orig)

        delta_y = (y_pert - y_base).detach()  # [B, T, H, W, C]
        f_lows.append(measure_f_low(delta_y))

    return np.array(f_lows)


# ── Main ────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUT_NPZ):
        log.info(f"Cache found: {OUT_NPZ}  — skipping computation")
        d = np.load(OUT_NPZ, allow_pickle=True)
        f_low = d["f_low"]
        rho, pval = float(d["rho"]), float(d["pval"])
    else:
        log.info("=== Building datasets & model ===")
        train_ds, val_ds, normalizer, model = build_all()
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=FWD_BATCH, shuffle=False)

        log.info("=== Computing FIM eigenvectors ===")
        grads = collect_grads(model, train_loader, normalizer, N_FIM)
        U, eigs = compute_eigenvectors(grads, K)

        log.info("=== Measuring f_low per FIM rank (2D spatial FFT) ===")
        f_low = measure_eigvec_f_low(model, U, val_loader, normalizer, N_TEST)

        rho, pval = spearmanr(np.arange(1, len(f_low) + 1), f_low)
        H  = min(train_ds[0][0].shape[-3], train_ds[0][0].shape[-2])   # approx
        np.savez(OUT_NPZ, f_low=f_low, rho=rho, pval=pval,
                 half=min(train_ds[0][1].shape[-3]//2, train_ds[0][1].shape[-2]//2))
        log.info(f"Saved → {OUT_NPZ}")

    ranks = np.arange(1, len(f_low) + 1)
    rho, pval = spearmanr(ranks, f_low)
    print(f"\n{'='*50}")
    print(f"FNO / Combustion (2D spatial FFT)")
    print(f"  f_low: {np.round(f_low, 4)}")
    print(f"  Spearman ρ = {rho:+.3f},  p = {pval:.3e}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
