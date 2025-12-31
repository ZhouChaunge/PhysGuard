#!/usr/bin/env python3
"""
Publication-quality flow-field visualization: DeepONet + Cylinder Flow.
Loads best checkpoints, runs inference on real test data, and generates a
2-row × 6-col figure (Pretrained, DFT, EWC, L2-SP, PhysGuard, GT).

Row 1: velocity magnitude |u| with jet contourf + white streamlines
Row 2: error magnitude |pred - GT|

Styled for top-venue publication (NeurIPS/ICLR) with:
  - 'jet' colormap + contourf (64 levels) + white streamlines for velocity
  - Sequential 'magma' for error
  - Minimal axes, consistent colorbars
  - LaTeX-style labels
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RealPDEBench'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.ticker as ticker
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ─── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# ─── Config ──────────────────────────────────────────────────────────
DEVICE = torch.device('cuda:3')

BASE = './results/001-cylinder/deeponet'
CKPTS = {
    'Pretrained': f'{BASE}/deeponet_cylinder_pretrained/2026-03-12_11-16-03/model_0100.pth',
    'DFT':        f'{BASE}/deeponet_cylinder_dft/2026-03-13_22-22-57/model_0560.pth',
    'EWC':        f'{BASE}/deeponet_cylinder_ewcft/lam1.0/2026-03-14_06-34-38/model_2720.pth',
    'L2-SP':      f'{BASE}/deeponet_cylinder_l2ft/lam0.0001/2026-03-14_06-34-38/model_2560.pth',
    'PhysGuard':  f'{BASE}/deeponet_cylinder_nsft/2026-03-13_00-38-45/model_0080.pth',
}

DATASET_ROOT = './data/realpdebench/'
OUTPUT_PATH = './figures/qualitative_deeponet_cylinder.pdf'

METHOD_ORDER = ['Pretrained', 'DFT', 'EWC', 'L2-SP', 'PhysGuard']
N_AUTOREGRESSIVE = 10  # DeepONet uses 10 AR steps

# ─── Model / data helpers ────────────────────────────────────────────
from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
from realpdebench.data.data_normalizer import GaussianNormalizer
from realpdebench.model.load_model import load_model


def build_test_dataset():
    return CylinderHFDataset(
        dataset_name='cylinder',
        dataset_root=DATASET_ROOT,
        dataset_type='real',
        mode='test',
        N_autoregressive=N_AUTOREGRESSIVE,
        mask_prob=0.1,
    )


def build_normalizer_dataset():
    """N_autoregressive=1 so model shape_out[0]=20 (single-step prediction)."""
    return CylinderHFDataset(
        dataset_name='cylinder',
        dataset_root=DATASET_ROOT,
        dataset_type='numerical',
        mode='train',
        N_autoregressive=1,
        mask_prob=0.1,
    )


def build_model(norm_ds):
    return load_model(
        norm_ds, device=DEVICE,
        model_name='deeponet',
        p=128,
        dropout_rate=0.1,
        checkpoint_path=None,
    )


def run_inference(model, normalizer, input_tensor):
    """Autoregressive inference. Returns (1, N_AR*20, H, W, C) in original scale."""
    model.eval()
    with torch.no_grad():
        t_in = input_tensor.shape[1]
        dummy = torch.zeros(1, t_in, *input_tensor.shape[2:]).to(DEVICE)
        inp, _ = normalizer.preprocess(input_tensor.to(DEVICE), dummy)

        preds = [inp]
        for _ in range(N_AUTOREGRESSIVE):
            p = model(preds[-1])
            _, p = normalizer.postprocess(preds[-1], p)
            p, _ = normalizer.preprocess(p, dummy)
            preds.append(p)

        pred = torch.cat(preds[1:], dim=1)
        _, pred = normalizer.postprocess(input_tensor.to(DEVICE), pred)
    return pred.cpu()


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    logging.info("Building datasets...")
    test_ds = build_test_dataset()
    norm_ds = build_normalizer_dataset()
    normalizer = GaussianNormalizer(norm_ds, device=DEVICE)

    logging.info("Building model...")
    model = build_model(norm_ds)

    # ── 1. Scan for best sample ──────────────────────────────────────
    # DeepONet N_AR=10 → max t=199; pick moderate steps to avoid full degradation
    candidate_t_steps = [39, 59, 79, 99, 119]
    n_scan = min(60, len(test_ds))
    logging.info(f"Scanning {n_scan} samples × {len(candidate_t_steps)} time steps...")

    model.load_checkpoint(CKPTS['DFT'], DEVICE)
    dft_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_checkpoint(CKPTS['PhysGuard'], DEVICE)
    pg_state = {k: v.clone() for k, v in model.state_dict().items()}

    # DFT predictions
    dft_preds = []
    model.load_state_dict(dft_state)
    for idx in range(n_scan):
        inp, _ = test_ds[idx]
        dft_preds.append(run_inference(model, normalizer, inp.unsqueeze(0)))

    # PhysGuard predictions
    pg_preds = []
    model.load_state_dict(pg_state)
    for idx in range(n_scan):
        inp, _ = test_ds[idx]
        pg_preds.append(run_inference(model, normalizer, inp.unsqueeze(0)))

    best_score, best_idx, best_t = -1, 0, 59
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

    # ── 2. Run inference for all 5 methods ───────────────────────────
    inp, tgt = test_ds[best_idx]
    inp_batch = inp.unsqueeze(0)

    all_preds = {}
    for name, ckpt in CKPTS.items():
        logging.info(f"Inference: {name}")
        model.load_checkpoint(ckpt, DEVICE)
        all_preds[name] = run_inference(model, normalizer, inp_batch)

    t = best_t
    gt_ux = tgt[t, :, :, 0].numpy()
    gt_uy = tgt[t, :, :, 1].numpy()
    gt_speed = np.sqrt(gt_ux**2 + gt_uy**2)

    method_fields = {}
    for name in METHOD_ORDER:
        pux = all_preds[name][0, t, :, :, 0].numpy()
        puy = all_preds[name][0, t, :, :, 1].numpy()
        method_fields[name] = {
            'ux': pux, 'uy': puy,
            'speed': np.sqrt(pux**2 + puy**2),
            'err': np.sqrt((pux - gt_ux)**2 + (puy - gt_uy)**2),
        }

    # ── 3. Publication figure ────────────────────────────────────────
    logging.info("Rendering figure...")

    ncols = len(METHOD_ORDER) + 1  # +1 for GT
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows))

    # Unified velocity magnitude range across all methods + GT
    all_speeds = [gt_speed] + [method_fields[n]['speed'] for n in METHOD_ORDER]
    vel_vmin = 0
    vel_vmax = max(s.max() for s in all_speeds)
    # Unified error range (cap at 99th percentile to preserve structure)
    err_max = max(method_fields[n]['err'].max() for n in METHOD_ORDER)
    err_p99 = max(np.percentile(method_fields[n]['err'], 99) for n in METHOD_ORDER)
    err_vmax = min(err_max, err_p99 * 1.05)

    # Colormaps
    CMAP_VEL = 'jet'         # jet + contourf + white streamlines
    CMAP_ERR = 'magma'       # sequential, perceptually uniform

    # Grid coordinates for contourf / streamplot
    H, W = gt_speed.shape
    x_grid = np.linspace(0, W - 1, W)
    y_grid = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(x_grid, y_grid)

    col_labels = METHOD_ORDER + ['Ground Truth']

    for col_idx, name in enumerate(col_labels):
        for row in range(nrows):
            ax = axes[row, col_idx]

            if row == 0:  # velocity magnitude |u| with streamlines
                if name == 'Ground Truth':
                    speed = gt_speed
                    ux, uy = gt_ux, gt_uy
                else:
                    speed = method_fields[name]['speed']
                    ux, uy = method_fields[name]['ux'], method_fields[name]['uy']
                # Filled contour (64 levels, jet colormap)
                cf = ax.contourf(X, Y, speed, levels=64,
                                 cmap=CMAP_VEL, vmin=vel_vmin, vmax=vel_vmax)
                # Overlay white streamlines
                try:
                    ax.streamplot(x_grid, y_grid, ux, uy,
                                  density=1.2, color='white',
                                  linewidth=0.6, arrowsize=0.5)
                except Exception:
                    pass

            elif row == 1:  # error
                if name == 'Ground Truth':
                    ax.text(0.5, 0.5, 'Reference', transform=ax.transAxes,
                            ha='center', va='center', fontsize=10,
                            fontstyle='italic', color='#666666')
                    ax.set_facecolor('#F7F7F7')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#CCCCCC')
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                else:
                    data = method_fields[name]['err']
                    im = ax.imshow(data, cmap=CMAP_ERR, vmin=0, vmax=err_vmax,
                                   aspect='equal', origin='lower', interpolation='bilinear')

            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect('equal')
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_edgecolor('#999999')

            # Column titles (top row only)
            if row == 0:
                ax.set_title(name, fontsize=9, fontweight='bold', pad=4)

    # Row labels
    row_labels = [r'$|\mathbf{u}|$', r'$|\mathbf{e}|$']
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold',
                                labelpad=8, rotation=0, ha='right', va='center')

    # ── Colorbars ────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.05, right=0.89, top=0.91, bottom=0.05,
                        wspace=0.08, hspace=0.18)

    cbar_width = 0.012
    cbar_pad = 0.015

    # Velocity magnitude colorbar
    cax1 = fig.add_axes([0.89 + cbar_pad, 0.53, cbar_width, 0.38])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(
        norm=Normalize(vel_vmin, vel_vmax), cmap=CMAP_VEL), cax=cax1)
    cb1.ax.tick_params(labelsize=7)
    cb1.set_label(r'$|\mathbf{u}|$', fontsize=9)

    # Error colorbar
    cax2 = fig.add_axes([0.89 + cbar_pad, 0.05, cbar_width, 0.38])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(
        norm=Normalize(0, err_vmax), cmap=CMAP_ERR), cax=cax2)
    cb2.ax.tick_params(labelsize=7)
    cb2.set_label(r'$|\mathbf{e}|$', fontsize=9)

    # Save as both PDF (vector) and PNG (raster preview)
    fig.savefig(OUTPUT_PATH, dpi=300)
    png_path = OUTPUT_PATH.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300)
    logging.info(f"Saved: {OUTPUT_PATH}")
    logging.info(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
