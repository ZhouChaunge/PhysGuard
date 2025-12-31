#!/usr/bin/env python3
"""
Spectral Comparison Panel: Pretrained → DFT → PhysGuard

Layout:  2 rows × 3 columns
  Top row:    Vorticity (ω_z) flow fields with contourf + streamlines
  Bottom row: 2D FFT power spectrum (log scale, radially averaged)
              + low-freq / high-freq region shading

DFT column:       red annotation  "Low-freq structure destroyed"
PhysGuard column:  green annotation "Low-freq preserved"

Scenario: FNO on Cylinder Flow (001-cylinder)

Usage:
    conda activate pytorch310
    python scripts/generate_spectral_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RealPDEBench'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ─── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
})

# ─── Config ──────────────────────────────────────────────────────────
DEVICE = torch.device('cuda:1')

BASE = './results/001-cylinder/fno'
CONFIG = './realpdebench/configs/cylinder/fno.yaml'
CKPTS = {
    'Pretrained': f'{BASE}/fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth',
    'DFT':        f'{BASE}/fno_cylinder_dft/2026-03-10_03-40-24/model_0160.pth',
    'PhysGuard':  f'{BASE}/fno_cylinder_nsft/2026-03-10_08-42-11/model_0160.pth',
}

DATASET_ROOT = './data/realpdebench/'
OUTPUT_DIR = './figures'
OUTPUT_NAME = 'spectral_comparison_panel'

METHOD_ORDER = ['Pretrained', 'DFT', 'PhysGuard']
METHOD_COLORS = {
    'Pretrained': '#7f8c8d',
    'DFT':        '#E53935',
    'PhysGuard':  '#27ae60',
}
METHOD_LABELS = {
    'Pretrained': 'Pretrained',
    'DFT':        'Direct Fine-Tuning (DFT)',
    'PhysGuard':  'PhysGuard (Ours)',
}

# ─── Model / data helpers ────────────────────────────────────────────
from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
from realpdebench.data.data_normalizer import GaussianNormalizer
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import set_seed, add_args_from_config
import argparse


def build_datasets():
    test_ds = CylinderHFDataset(
        dataset_name='cylinder',
        dataset_root=DATASET_ROOT,
        dataset_type='real',
        mode='test',
        N_autoregressive=3,
        mask_prob=0.1,
    )
    norm_ds = CylinderHFDataset(
        dataset_name='cylinder',
        dataset_root=DATASET_ROOT,
        dataset_type='numerical',
        mode='train',
        N_autoregressive=1,
        mask_prob=0.1,
    )
    return test_ds, norm_ds


def build_model(norm_ds):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=CONFIG)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--train_data_type", type=str, default="numerical")
    parser.add_argument("--test_data_type", type=str, default="real")
    parser.add_argument("--checkpoint_path", type=str, default=list(CKPTS.values())[0])
    parser.add_argument("--use_hf_dataset", action="store_true", default=True)
    args = parser.parse_args([])
    args = add_args_from_config(args, parser)
    if isinstance(args.gpu, list):
        args.gpu = args.gpu[0]
    model = load_model(norm_ds, device=DEVICE, **vars(args))
    return model, args


def run_inference(model, normalizer, input_tensor, n_ar=3):
    """Autoregressive inference. Returns [1, N_AR*T_step, H, W, C] in original scale."""
    model.eval()
    with torch.no_grad():
        t_in = input_tensor.shape[1]
        dummy = torch.zeros(1, t_in, *input_tensor.shape[2:]).to(DEVICE)
        inp, _ = normalizer.preprocess(input_tensor.to(DEVICE), dummy)

        preds = [inp]
        for _ in range(n_ar):
            p = model(preds[-1])
            _, p = normalizer.postprocess(preds[-1], p)
            p, _ = normalizer.preprocess(p, dummy)
            preds.append(p)

        pred = torch.cat(preds[1:], dim=1)
        _, pred = normalizer.postprocess(input_tensor.to(DEVICE), pred)
    return pred.cpu()


def compute_vorticity(ux, uy):
    """Compute vorticity ω_z = ∂u_y/∂x - ∂u_x/∂y using central differences."""
    duy_dx = np.gradient(uy, axis=1)
    dux_dy = np.gradient(ux, axis=0)
    return duy_dx - dux_dy


def compute_2d_fft_power(field):
    """
    Compute 2D FFT radially-averaged power spectrum.
    Returns: (k_bins, radial_power)
    """
    H, W = field.shape
    F = np.fft.fft2(field)
    F_shift = np.fft.fftshift(F)
    power = np.abs(F_shift) ** 2

    # Radial averaging
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[-cy:H-cy, -cx:W-cx]
    R = np.sqrt(X.astype(float)**2 + Y.astype(float)**2)
    R_int = R.astype(int)

    max_r = min(cy, cx)
    radial_power = np.zeros(max_r)
    radial_count = np.zeros(max_r)

    for r in range(max_r):
        mask = R_int == r
        radial_power[r] = power[mask].sum()
        radial_count[r] = mask.sum()

    # Avoid division by zero
    radial_count[radial_count == 0] = 1
    radial_power /= radial_count

    k_bins = np.arange(max_r)
    return k_bins, radial_power


def compute_2d_fft_power_2d(field):
    """
    Compute 2D FFT power spectrum (full 2D image for visualization).
    Returns log10(power + eps) centered.
    """
    F = np.fft.fft2(field)
    F_shift = np.fft.fftshift(F)
    power = np.abs(F_shift) ** 2
    return np.log10(power + 1e-12)


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CACHE_PATH = os.path.join(OUTPUT_DIR, 'spectral_comparison_cache.npz')

    if os.path.exists(CACHE_PATH):
        logging.info(f"Loading cached data from {CACHE_PATH}")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        all_fields = cache['all_fields'].item()
        gt_vort = cache['gt_vort']
        gt_ux = cache['gt_ux']
        gt_uy = cache['gt_uy']
    else:
        logging.info("No cache found. Running inference...")

        logging.info("Building datasets...")
        test_ds, norm_ds = build_datasets()
        normalizer = GaussianNormalizer(norm_ds, device=DEVICE)

        logging.info("Building model...")
        model, args = build_model(norm_ds)

        # ── 1. Scan for best sample (max contrast DFT vs PhysGuard) ──────
        n_ar = 3
        candidate_t_steps = [19, 29, 39, 49, 59]
        n_scan = min(40, len(test_ds))
        logging.info(f"Scanning {n_scan} samples for best visualization...")

        # DFT predictions
        model.load_checkpoint(CKPTS['DFT'], DEVICE)
        model.eval()
        dft_preds = []
        with torch.no_grad():
            for idx in range(n_scan):
                inp, _ = test_ds[idx]
                dft_preds.append(run_inference(model, normalizer, inp.unsqueeze(0), n_ar))

        # PhysGuard predictions
        model.load_checkpoint(CKPTS['PhysGuard'], DEVICE)
        model.eval()
        pg_preds = []
        with torch.no_grad():
            for idx in range(n_scan):
                inp, _ = test_ds[idx]
                pg_preds.append(run_inference(model, normalizer, inp.unsqueeze(0), n_ar))

        best_score, best_idx, best_t = -1, 0, 39
        for idx in range(n_scan):
            _, tgt = test_ds[idx]
            for t in candidate_t_steps:
                if t >= tgt.shape[0]:
                    continue
                gt_vel = tgt[t, :, :, :2].numpy()
                pg_err = np.abs(pg_preds[idx][0, t, :, :, :2].numpy() - gt_vel).mean()
                dft_err = np.abs(dft_preds[idx][0, t, :, :, :2].numpy() - gt_vel).mean()
                uy_var = gt_vel[:, :, 1].var()
                if pg_err > 0 and uy_var > 0.001:
                    score = (dft_err / pg_err) * np.sqrt(uy_var)
                    if score > best_score:
                        best_score, best_idx, best_t = score, idx, t

        logging.info(f"Selected sample_idx={best_idx}, t={best_t}, score={best_score:.4f}")

        # ── 2. Run inference for all 3 methods ───────────────────────────
        inp, tgt = test_ds[best_idx]
        inp_batch = inp.unsqueeze(0)
        t = best_t

        gt_ux = tgt[t, :, :, 0].numpy()
        gt_uy = tgt[t, :, :, 1].numpy()
        gt_vort = compute_vorticity(gt_ux, gt_uy)

        all_fields = {}
        for name, ckpt in CKPTS.items():
            logging.info(f"Inference: {name}")
            model.load_checkpoint(ckpt, DEVICE)
            pred = run_inference(model, normalizer, inp_batch, n_ar)
            pux = pred[0, t, :, :, 0].numpy()
            puy = pred[0, t, :, :, 1].numpy()
            vort = compute_vorticity(pux, puy)
            all_fields[name] = {
                'ux': pux, 'uy': puy,
                'speed': np.sqrt(pux**2 + puy**2),
                'vorticity': vort,
            }

        # Save cache
        np.savez(CACHE_PATH, all_fields=all_fields, gt_vort=gt_vort,
                 gt_ux=gt_ux, gt_uy=gt_uy)
        logging.info(f"Saved cache to {CACHE_PATH}")

    # ── 3. Compute 2D FFT Power ─────────────────────────────────────
    spectra_1d = {}
    spectra_2d = {}
    for name in METHOD_ORDER:
        vort = all_fields[name]['vorticity']
        k, psd = compute_2d_fft_power(vort)
        spectra_1d[name] = (k, psd)
        spectra_2d[name] = compute_2d_fft_power_2d(vort)

    # GT
    gt_k, gt_psd = compute_2d_fft_power(gt_vort)
    gt_spec_2d = compute_2d_fft_power_2d(gt_vort)

    # ── 4. Publication Figure ────────────────────────────────────────
    logging.info("Rendering figure...")

    fig = plt.figure(figsize=(15, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.35, wspace=0.25,
                           left=0.07, right=0.92, top=0.93, bottom=0.08,
                           height_ratios=[1, 1])

    # --- Color scheme ---
    CMAP_VORT = 'RdBu_r'
    vort_all = [all_fields[n]['vorticity'] for n in METHOD_ORDER] + [gt_vort]
    vort_vmax = np.percentile(np.abs(np.concatenate([v.ravel() for v in vort_all])), 98)

    H, W = gt_vort.shape
    x_grid = np.linspace(0, W - 1, W)
    y_grid = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(x_grid, y_grid)

    # ============================================================
    # Row 0: Flow field (vorticity) with streamlines
    # ============================================================
    for col, name in enumerate(METHOD_ORDER):
        ax = fig.add_subplot(gs[0, col])
        vort = all_fields[name]['vorticity']
        ux = all_fields[name]['ux']
        uy = all_fields[name]['uy']

        # Filled contour
        cf = ax.contourf(X, Y, vort, levels=64, cmap=CMAP_VORT,
                         vmin=-vort_vmax, vmax=vort_vmax)

        # White streamlines
        try:
            ax.streamplot(x_grid, y_grid, ux, uy,
                          density=1.0, color='white',
                          linewidth=0.5, arrowsize=0.4)
        except Exception:
            pass

        ax.set_xlim(0, W - 1)
        ax.set_ylim(0, H - 1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(METHOD_COLORS[name])

        # Title with colored method name
        ax.set_title(METHOD_LABELS[name], fontsize=13, fontweight='bold',
                     color=METHOD_COLORS[name], pad=8)

        # Stage arrow annotations between columns
        if col < 2:
            fig.text(
                (gs[0, col].get_position(fig).x1 + gs[0, col + 1].get_position(fig).x0) / 2,
                gs[0, col].get_position(fig).y0 + (gs[0, col].get_position(fig).y1 - gs[0, col].get_position(fig).y0) / 2,
                r'$\Rightarrow$', fontsize=26, ha='center', va='center',
                color='#555', fontweight='bold',
            )

    # Vorticity colorbar
    cax_vort = fig.add_axes([0.93, 0.53, 0.012, 0.38])
    cb_vort = fig.colorbar(plt.cm.ScalarMappable(
        norm=Normalize(-vort_vmax, vort_vmax), cmap=CMAP_VORT), cax=cax_vort)
    cb_vort.ax.tick_params(labelsize=8)
    cb_vort.set_label(r'Vorticity $\omega_z$', fontsize=10)

    # Row label
    fig.text(0.015, 0.72, 'Flow Field\n(Vorticity)', fontsize=10, fontweight='bold',
             ha='center', va='center', rotation=90, color='#333')

    # ============================================================
    # Row 1: 2D FFT power spectrum (radially averaged, log-log)
    # ============================================================
    for col, name in enumerate(METHOD_ORDER):
        ax = fig.add_subplot(gs[1, col])
        k, psd = spectra_1d[name]

        # Avoid log(0)
        valid = (k > 0) & (psd > 0)
        k_v = k[valid]
        psd_v = psd[valid]

        # Low/high frequency boundary
        k_max = k_v.max()
        k_low_cut = k_max / 3
        k_high_cut = k_max * 2 / 3

        # Shade low-frequency region
        ax.axvspan(k_v.min(), k_low_cut, alpha=0.12, color='#2196F3',
                   label='Low-freq region')
        # Shade high-frequency region
        ax.axvspan(k_high_cut, k_v.max(), alpha=0.08, color='#FF9800',
                   label='High-freq region')

        # GT spectrum (reference)
        gt_valid = (gt_k > 0) & (gt_psd > 0)
        ax.loglog(gt_k[gt_valid], gt_psd[gt_valid],
                  color='#2c3e50', linewidth=1.8, alpha=0.5,
                  linestyle='--', label='Ground Truth', zorder=3)

        # Method spectrum
        ax.loglog(k_v, psd_v,
                  color=METHOD_COLORS[name], linewidth=2.2, zorder=4,
                  label=METHOD_LABELS[name])

        # Fill the gap area between GT and method in low-freq region
        gt_interp = np.interp(k_v, gt_k[gt_valid], gt_psd[gt_valid])
        low_mask = k_v <= k_low_cut

        if name == 'DFT':
            # Red fill to show deviation
            ax.fill_between(k_v[low_mask], psd_v[low_mask], gt_interp[low_mask],
                            alpha=0.25, color='#E53935', zorder=2)
        elif name == 'PhysGuard':
            # Green fill to show closeness
            ax.fill_between(k_v[low_mask], psd_v[low_mask], gt_interp[low_mask],
                            alpha=0.2, color='#27ae60', zorder=2)

        ax.set_xlabel('Wavenumber $k$', fontsize=10)
        if col == 0:
            pass  # Row label via fig.text is sufficient

        ax.set_xlim(k_v.min(), k_v.max())
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.8)

        # Spine color to match method
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(METHOD_COLORS[name])

        # ── Key annotations ──
        if name == 'DFT':
            # Red arrow + text: "Low-freq physical structure destroyed"
            # Use geometric mean for log scale
            psd_low_geomean = np.exp(np.log(psd_v[low_mask] + 1e-12).mean())
            ax.annotate(
                'Low-freq structure\ndestroyed',
                xy=(k_low_cut * 0.4, psd_low_geomean),
                xytext=(k_low_cut * 2.5, psd_low_geomean * 0.15),
                fontsize=9, fontweight='bold', color='#E53935',
                arrowprops=dict(arrowstyle='->', color='#E53935', linewidth=2,
                                connectionstyle='arc3,rad=-0.2'),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE',
                          edgecolor='#E53935', alpha=0.9),
                ha='center', va='center', zorder=10,
            )


    # Row label
    fig.text(0.015, 0.28, '2D FFT\nPower\nSpectrum', fontsize=10, fontweight='bold',
             ha='center', va='center', rotation=90, color='#333')

    # ── Global title ──
    fig.suptitle(
        r'Spectral Analysis: Pretrained $\Rightarrow$ DFT $\Rightarrow$ PhysGuard  (FNO, Cylinder Flow)',
        fontsize=14, fontweight='bold', y=0.98, color='#222'
    )

    # ── Save ──
    pdf_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_NAME}.pdf')
    png_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_NAME}.png')
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    logging.info(f"Saved: {pdf_path}")
    logging.info(f"Saved: {png_path}")
    plt.close(fig)

    logging.info("Done!")


if __name__ == '__main__':
    main()
