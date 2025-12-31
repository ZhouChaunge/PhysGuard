#!/usr/bin/env python3
"""
RQ1 NeurIPS Proof Figure: FIM Principal Subspace ↔ Low-Frequency Physics
=========================================================================
Single architecture (FNO), 3 datasets, unified evidence.

Figure layout (2 rows × 3 columns):
  Row 1 (a,b,c): Radial PSD of Δy — FIM rank 1 vs Random direction
  Row 2 (d,e,f): f_low vs FIM eigenvalue rank + random baseline + Spearman ρ

Evidence structure:
  Sufficient:   FIM rank 1 produces low-frequency Δy  (Row 1, blue curves)
  Necessary:    Random direction does NOT concentrate in low-freq  (Row 1, red curves)
  Systematic:   f_low decreases monotonically with rank  (Row 2, Spearman ρ < 0)
  General:      True across 3 physically distinct datasets  (3 columns)

Run:
  cd RealPDEBench
  CUDA_VISIBLE_DEVICES=0 python \
      -u ../scripts/rq1_neurips_proof.py
"""

import os, sys, importlib, logging
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.join(SCRIPT_DIR, "..", "RealPDEBench")
sys.path.insert(0, REPO_DIR)

DATASET_ROOT = "./data/realpdebench/"
RESULTS_ROOT = "./results"
OUTPUT_DIR   = "./figures"
CACHE_DIR    = "/tmp"

# ── Shared hyper-parameters ────────────────────────────────────────
TARGET_PARAM = "spectral_convs.1.weights1"   # FNO Layer 2

N_FIM     = 80       # gradient samples for empirical FIM
K         = 50       # number of FIM eigenvectors to extract
N_TEST    = 40       # test samples for Δy
N_RANDOM  = 10       # random baseline directions
EPSILON   = 1e-3     # perturbation step size
FWD_BATCH = 4        # forward-pass mini-batch size
SEED      = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset configurations ─────────────────────────────────────────
DATASETS = OrderedDict([
    ('cylinder', dict(
        title='Cylinder Flow\n(laminar)',
        ds_import='realpdebench.data.fluid_hf_dataset',
        ds_class='CylinderHFDataset',
        ds_name='cylinder',
        ckpt=f'{RESULTS_ROOT}/001-cylinder/fno/fno_cylinder_pretrained'
             '/2026-03-09_17-33-29/model_3760.pth',
        fno=dict(model_name='fno', modes1=4, modes2=12, modes3=16,
                 n_layers=4, width=64),
    )),
    ('ctrl_cylinder', dict(
        title='Controlled Cylinder\n(transitional)',
        ds_import='realpdebench.data.fluid_hf_dataset',
        ds_class='ControlledCylinderHFDataset',
        ds_name='controlled_cylinder',
        ckpt=f'{RESULTS_ROOT}/002-control_cylinder/fno/fno_control_pretrained'
             '/2026-03-15_02-36-15/model_3840.pth',
        fno=dict(model_name='fno', modes1=4, modes2=12, modes3=16,
                 n_layers=4, width=64),
    )),
    ('combustion', dict(
        title='Turbulent Combustion\n(reactive)',
        ds_import='realpdebench.data.combustion_hf_dataset',
        ds_class='CombustionHFDataset',
        ds_name='combustion',
        ckpt=f'{RESULTS_ROOT}/003-combustion/fno/fno_combustion_pretrained'
             '/2026-03-20_11-54-28/model_2000.pth',
        fno=dict(model_name='fno', modes1=4, modes2=16, modes3=16,
                 n_layers=4, width=64),
    )),
])


# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def build_model_and_data(cfg):
    """Load dataset, normalizer, and pretrained FNO model."""
    mod = importlib.import_module(cfg['ds_import'])
    DSClass = getattr(mod, cfg['ds_class'])
    from realpdebench.data.data_normalizer import GaussianNormalizer
    from realpdebench.model.load_model import load_model

    common = dict(dataset_name=cfg['ds_name'], dataset_root=DATASET_ROOT)
    train_ds = DSClass(mode='train', dataset_type='numerical', **common)
    val_ds   = DSClass(mode='val',   dataset_type='real',      **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)

    model = load_model(train_ds, device=DEVICE, **cfg['fno'])
    ckpt  = torch.load(cfg['ckpt'], map_location=DEVICE)
    raw   = ckpt.get('model_state_dict', ckpt)
    state = {k.replace('module.', ''): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return train_ds, val_ds, normalizer, model


def collect_grads(model, loader, normalizer, n):
    """Collect per-sample gradients for empirical FIM."""
    grads = []
    pbar  = tqdm(total=n, desc="  FIM grads")
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
    """FIM top-k eigenvectors via Gram matrix trick: K = J Jᵀ."""
    J = torch.stack(grads, dim=0)        # [N_FIM, d]
    G = J @ J.T                           # [N_FIM, N_FIM]
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
    return torch.stack(U, dim=1), L[:k_act]   # [d, k_act]


def get_base_outputs(model, loader, normalizer, n):
    """Unperturbed model outputs on n test samples."""
    xs, ys = [], []
    cnt = 0
    with torch.no_grad():
        for x, y in loader:
            if cnt >= n:
                break
            x, _ = normalizer.preprocess(x, y)
            rem = min(n - cnt, x.shape[0])
            x = x[:rem]
            xs.append(x.cpu())
            ys.append(model(x.to(DEVICE)).cpu())
            cnt += rem
    return torch.cat(xs), torch.cat(ys)


# ── 2D Spatial FFT → Radial PSD ──────────────────────────────────

def compute_2d_psd(dy: torch.Tensor):
    """dy: [B, T, H, W, C] → (psd_1d [half], half)."""
    B, T, H, W, C = dy.shape
    half = min(H // 2, W // 2)

    dy_avg = dy.float().mean(dim=1)                      # [B, H, W, C]
    F      = torch.fft.fftn(dy_avg, dim=[1, 2])          # [B, H, W, C]
    P      = F.abs() ** 2
    P_half = P[:, :H // 2, :W // 2, :]

    ii = torch.arange(H // 2).float()
    jj = torch.arange(W // 2).float()
    gi, gj = torch.meshgrid(ii, jj, indexing='ij')
    radial = torch.floor(torch.sqrt(gi ** 2 + gj ** 2)).long()
    valid  = radial < half

    r_flat  = radial[valid]
    P_bc    = P_half.permute(0, 3, 1, 2).reshape(B * C, H // 2, W // 2)
    P_sum   = P_bc.sum(dim=0)
    P_valid = P_sum[valid]

    psd = torch.zeros(half)
    psd.scatter_add_(0, r_flat, P_valid)
    cnt = torch.zeros(half, dtype=torch.long)
    cnt.scatter_add_(0, r_flat, torch.ones_like(r_flat))
    cnt = cnt.clamp(min=1)
    psd = (psd / (cnt.float() * B * C)).numpy()
    return psd, half


def measure_f_low(psd, half):
    """Low-frequency energy fraction: Σ PSD(k<k_th) / Σ PSD."""
    k_th = max(1, round(half / 3))
    return float(psd[:k_th].sum() / (psd.sum() + 1e-15))


def perturb_and_measure(model, u_vec, x_test, y_base, eps=EPSILON):
    """Perturb TARGET_PARAM by ε·u, measure Δy, return (psd, f_low, half)."""
    for pname, p in model.named_parameters():
        if pname != TARGET_PARAM:
            continue
        # Build complex or real delta
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
        fl = measure_f_low(psd, half)
        return psd, fl, half

    raise ValueError(f"Parameter '{TARGET_PARAM}' not found.")


# ═══════════════════════════════════════════════════════════════════
#  PER-DATASET PROCESSING
# ═══════════════════════════════════════════════════════════════════

def process_dataset(ds_key, cfg):
    """Compute PSD + f_low for FIM ranks 1..K and random baselines."""
    cache_path = os.path.join(CACHE_DIR, f'rq1_neurips_{ds_key}.npz')

    if os.path.exists(cache_path):
        log.info(f"[{ds_key}] Cache found → {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        results = {}
        for k in data.files:
            v = data[k]
            results[k] = float(v) if v.ndim == 0 else v
        return results

    log.info(f"[{ds_key}] Loading model and data …")
    train_ds, val_ds, normalizer, model = build_model_and_data(cfg)

    loader_fim  = DataLoader(train_ds, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    loader_test = DataLoader(val_ds, batch_size=FWD_BATCH, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── FIM eigenvectors ──
    log.info(f"[{ds_key}] FIM eigenvectors (N_FIM={N_FIM}, K={K}) …")
    grads = collect_grads(model, loader_fim, normalizer, N_FIM)
    U, eigvals = top_k_eigenvectors(grads, K)
    k_act = U.shape[1]
    log.info(f"[{ds_key}]   λ₁={eigvals[0]:.3e}  λ{k_act}={eigvals[-1]:.3e}  "
             f"k_act={k_act}")

    # ── Base outputs ──
    log.info(f"[{ds_key}] Base outputs (N_TEST={N_TEST}) …")
    x_test, y_base = get_base_outputs(model, loader_test, normalizer, N_TEST)
    log.info(f"[{ds_key}]   y shape: {tuple(y_base.shape)}")

    results = {}

    # ── FIM ranks 1..k_act ──
    f_low_ranks = []
    for j in range(k_act):
        u_j = U[:, j]
        psd_j, fl_j, half = perturb_and_measure(model, u_j, x_test, y_base)
        f_low_ranks.append(fl_j)
        if j == 0:
            results['psd_rank1'] = psd_j
        if j == k_act - 1:
            results['psd_rankK'] = psd_j
        if (j + 1) % 10 == 0 or j == 0:
            log.info(f"[{ds_key}]   Rank {j+1:3d}: f_low = {fl_j:.4f}")

    results['f_low_ranks'] = np.array(f_low_ranks)
    results['half'] = half

    # ── Random directions ──
    log.info(f"[{ds_key}] Random baseline ({N_RANDOM} directions) …")
    f_low_rand = []
    psd_rand_list = []
    param_dim = U.shape[0]
    for i in range(N_RANDOM):
        torch.manual_seed(SEED + 1000 + i)
        r = torch.randn(param_dim)
        r /= (r.norm() + 1e-12)
        psd_r, fl_r, _ = perturb_and_measure(model, r, x_test, y_base)
        f_low_rand.append(fl_r)
        psd_rand_list.append(psd_r)
        log.info(f"[{ds_key}]   Random {i+1:2d}: f_low = {fl_r:.4f}")

    results['f_low_random'] = np.array(f_low_rand)
    results['psd_random']   = np.mean(psd_rand_list, axis=0)

    # ── Spearman correlation ──
    ranks_arr = np.arange(1, len(f_low_ranks) + 1)
    rho, pval = spearmanr(ranks_arr, f_low_ranks)
    results['rho']  = rho
    results['pval'] = pval
    log.info(f"[{ds_key}]   Spearman ρ = {rho:.4f}, p = {pval:.2e}")

    # ── Cache ──
    np.savez(cache_path, **results)
    log.info(f"[{ds_key}] Cached → {cache_path}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════

def make_figure(all_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.stats import mannwhitneyu

    # ── NeurIPS-compatible style (textwidth ≈ 5.5in) ──
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    })

    C_FIM  = '#1565C0'
    C_RAND = '#C62828'
    C_BAND = '#90CAF9'

    fig = plt.figure(figsize=(5.5, 3.6))
    gs  = GridSpec(2, 3, hspace=0.62, wspace=0.35,
                   left=0.10, right=0.995, top=0.91, bottom=0.12)

    ds_keys = list(DATASETS.keys())
    titles = {'cylinder': 'Cylinder (laminar)',
              'ctrl_cylinder': 'Ctrl-Cylinder (transitional)',
              'combustion': 'Combustion (reactive)'}

    for col, ds_key in enumerate(ds_keys):
        res = all_results[ds_key]
        half = int(res['half'])
        k_th = max(1, round(half / 3))
        ks   = np.arange(half)

        # ────────── Row 1: PSD of Δy ──────────
        ax = fig.add_subplot(gs[0, col])

        psd1  = res['psd_rank1']
        psd_r = res['psd_random']
        fl1   = measure_f_low(psd1, half)
        fl_r  = measure_f_low(psd_r, half)

        psd1_n = psd1 / (psd1.sum() + 1e-15)
        psd_r_n = psd_r / (psd_r.sum() + 1e-15)

        ax.plot(ks, psd1_n, color=C_FIM, lw=1.6,
                label=f'FIM top-1 ($f_{{\\rm low}}\\!={fl1:.3f}$)')
        ax.plot(ks, psd_r_n, color=C_RAND, lw=1.3, ls='-.',
                label=f'Random ($f_{{\\rm low}}\\!={fl_r:.3f}$)')

        ax.axvspan(0, k_th - 0.5, alpha=0.10, color=C_BAND)
        ax.axvline(k_th - 0.5, color=C_BAND, lw=0.8, ls=':', alpha=0.7)

        ax.set_yscale('log')
        if col == 0:
            ax.set_ylabel('Energy density (norm.)')
        ax.set_xlabel('Radial wavenumber $k$')
        ax.set_title(titles[ds_key], fontsize=7.5, pad=3)
        ax.legend(loc='best', framealpha=0.92,
                  borderpad=0.3, handlelength=1.0, fontsize=5.5)
        ax.set_xlim(0, half - 1)
        ax.set_ylim(bottom=1e-6)
        ax.grid(axis='y', alpha=0.2, lw=0.4)

        # ────────── Row 2: Violin — FIM group vs Random group ──────────
        ax2 = fig.add_subplot(gs[1, col])

        fl_fim  = res['f_low_ranks']    # shape (50,)
        fl_rand = res['f_low_random']   # shape (10,)

        # Violin plot
        parts = ax2.violinplot(
            [fl_fim, fl_rand], positions=[1, 2], widths=0.6,
            showmeans=True, showmedians=False, showextrema=False)

        # Style the violins
        for i, body in enumerate(parts['bodies']):
            color = C_FIM if i == 0 else C_RAND
            body.set_facecolor(color)
            body.set_alpha(0.25)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)
        parts['cmeans'].set_colors([C_FIM, C_RAND])
        parts['cmeans'].set_linewidths(1.5)

        # Overlay individual points (jittered)
        np.random.seed(42)
        jitter_fim  = 1 + np.random.uniform(-0.12, 0.12, len(fl_fim))
        jitter_rand = 2 + np.random.uniform(-0.12, 0.12, len(fl_rand))
        ax2.scatter(jitter_fim, fl_fim, color=C_FIM, s=4, alpha=0.5,
                    edgecolors='none', zorder=3)
        ax2.scatter(jitter_rand, fl_rand, color=C_RAND, s=8, alpha=0.6,
                    edgecolors='none', zorder=3)

        # Y-axis top reference
        y_top = max(fl_fim.max(), fl_rand.max()) + 0.015

        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['FIM\neigenvec.', 'Random\ndir.'],
                            fontsize=6)
        if col == 0:
            ax2.set_ylabel('$f_{\\rm low}$')

        # Y-axis range: show both groups clearly
        all_vals = np.concatenate([fl_fim, fl_rand])
        y_lo = all_vals.min() - 0.03
        y_hi = y_top + 0.04
        y_lo = max(0.0, y_lo)
        y_hi = min(1.05, y_hi)
        ax2.set_ylim(y_lo, y_hi)
        ax2.set_xlim(0.3, 2.7)
        ax2.grid(axis='y', alpha=0.2, lw=0.4)

    for ext in ('pdf', 'png'):
        path = os.path.join(OUTPUT_DIR, f'rq1_neurips_proof.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        log.info(f'Saved → {path}')
    plt.close(fig)
    log.info('Figure complete.')


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info('=' * 70)
    log.info('RQ1 NeurIPS Proof: FIM subspace ↔ low-frequency physics')
    log.info(f'  Datasets   : {list(DATASETS.keys())}')
    log.info(f'  TARGET     : {TARGET_PARAM}')
    log.info(f'  N_FIM={N_FIM}  K={K}  N_TEST={N_TEST}  '
             f'N_RANDOM={N_RANDOM}  ε={EPSILON}')
    log.info('=' * 70)

    all_results = {}
    for ds_key, cfg in DATASETS.items():
        all_results[ds_key] = process_dataset(ds_key, cfg)

    log.info('Generating figure …')
    make_figure(all_results)
    log.info('All done.')


if __name__ == '__main__':
    main()
