"""
Spectrum analysis for FNO on Cylinder Flow (5 methods + GT).

Generates figures/004-spectrum.png with:
  (a) Frequency-band fRMSE bar chart (Low / Mid / High)
  (b) Spherical shell PSD curves

Also saves raw data to figures/004-spectrum_raw_data.pt

Uses the EXACT same 3D FFT + spherical shell binning as
RealPDEBench/realpdebench/utils/metrics.py  eval_metrics(),
with global sqrt(mean) over the full dataset (not per-batch).
"""

import os, sys, math
import numpy as np
import torch
import tqdm
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RealPDEBench'))

from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
from realpdebench.data.data_normalizer import GaussianNormalizer
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import set_seed, add_args_from_config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Config
# ============================================================================
BASE = './results/001-cylinder/fno'
CONFIG = './realpdebench/configs/cylinder/fno.yaml'
DATASET_ROOT = './data/realpdebench/'
OUTPUT_DIR = './figures'
GPU = 1

METHODS = {
    'Pretrained': f'{BASE}/fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth',
    'DFT':        f'{BASE}/fno_cylinder_dft/2026-03-10_03-40-24/model_0160.pth',
    'L2-SP':      f'{BASE}/fno_cylinder_l2ft/lam0.0001/2026-03-14_11-45-57/model_0160.pth',
    'EWC':        f'{BASE}/fno_cylinder_ewcft/lam0.1/2026-03-13_22-23-43/model_0160.pth',
    'PhysGuard':  f'{BASE}/fno_cylinder_nsft/2026-03-10_08-42-11/model_0160.pth',
}

METHOD_COLORS = {
    'Pretrained': '#7f8c8d',
    'DFT':        '#E53935',
    'L2-SP':      '#FB8C00',
    'EWC':        '#8E24AA',
    'PhysGuard':  '#1E88E5',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
})


# ============================================================================
# 3D FFT + spherical shell binning  (identical to metrics.py eval_metrics)
# ============================================================================

def build_shell_map(t, h, w):
    """Precompute (i,j,k)->shell_index mapping, vectorised."""
    n_shells = min(t // 2, h // 2, w // 2)
    ii = torch.arange(t // 2)
    jj = torch.arange(h // 2)
    kk = torch.arange(w // 2)
    I, J, K = torch.meshgrid(ii, jj, kk, indexing='ij')
    shell_idx = torch.floor(torch.sqrt(I.float()**2 + J.float()**2 + K.float()**2)).long()
    valid = shell_idx <= n_shells - 1
    flat_idx = shell_idx[valid]
    shell_count = torch.zeros(n_shells)
    for s in range(n_shells):
        shell_count[s] = (flat_idx == s).sum().item()
    return {
        'flat_idx': flat_idx,
        'I_valid': I[valid], 'J_valid': J[valid], 'K_valid': K[valid],
        'shell_count': shell_count, 'n_shells': n_shells,
    }


def shell_bin_batch(pred, target, smap):
    """
    For one DataLoader batch, return shell-binned |err|^2 and |target|^2
    SUMMED over the batch dimension (not averaged), so the caller can
    accumulate across batches and do a single global mean at the end.

    pred, target: [B, T, H, W, C] CPU tensors
    Returns: err_sum [n_shells, C], norm_sum [n_shells, C],
             pred_pow_sum [n_shells, C], batch_size int
    """
    b, t, h, w, c = pred.shape
    n = smap['n_shells']
    fi = smap['flat_idx']
    Iv, Jv, Kv = smap['I_valid'], smap['J_valid'], smap['K_valid']

    pred_F = torch.fft.fftn(pred, dim=[1, 2, 3])
    tgt_F  = torch.fft.fftn(target, dim=[1, 2, 3])

    err2 = torch.abs(pred_F - tgt_F) ** 2          # [B,T,H,W,C]
    norm2 = torch.abs(tgt_F) ** 2
    ppow2 = torch.abs(pred_F) ** 2
    del pred_F, tgt_F

    idx_b = fi.unsqueeze(0).unsqueeze(-1).expand(b, -1, c)

    err_vals  = err2[:, Iv, Jv, Kv, :]             # [B, N_valid, C]
    norm_vals = norm2[:, Iv, Jv, Kv, :]
    ppow_vals = ppow2[:, Iv, Jv, Kv, :]
    del err2, norm2, ppow2

    err_F  = torch.zeros(b, n, c).scatter_add_(1, idx_b, err_vals)
    norm_F = torch.zeros(b, n, c).scatter_add_(1, idx_b, norm_vals)
    ppow_F = torch.zeros(b, n, c).scatter_add_(1, idx_b, ppow_vals)

    return err_F.sum(0), norm_F.sum(0), ppow_F.sum(0), b


class SpectralAccumulator3D:
    """
    Accumulates raw shell-binned |err|^2 sums across the full dataset,
    then computes sqrt(global_mean)/THW at the end — identical to
    eval_metrics() being called with the full concatenated tensor.
    """

    def __init__(self):
        self.err_sum = None       # [n_shells, C]
        self.norm_sum = None
        self.pred_pow_sum = None
        self.n_total = 0
        self.n_shells = None
        self._smap = None
        self._thw = None

    def accumulate(self, pred, target):
        """pred, target: [B, T, H, W, C] CPU tensors."""
        _, t, h, w, _ = pred.shape
        if self._smap is None:
            self._smap = build_shell_map(t, h, w)
            self._thw = t * h * w
            self.n_shells = self._smap['n_shells']

        es, ns, ps, b = shell_bin_batch(pred, target, self._smap)
        if self.err_sum is None:
            self.err_sum = es
            self.norm_sum = ns
            self.pred_pow_sum = ps
        else:
            self.err_sum += es
            self.norm_sum += ns
            self.pred_pow_sum += ps
        self.n_total += b

    # ---------- metrics (call after all batches) ----------

    def get_frmse(self):
        """Global sqrt(mean(err_F))/THW  then band-average over shells+channels."""
        _err = torch.sqrt(self.err_sum / self.n_total) / self._thw   # [n_shells, C]
        n = self.n_shells
        iL = int(np.round(n / 3))
        iH = int(np.round(n * 2 / 3))
        return {
            'low':  _err[:iL].mean().item(),
            'mid':  _err[iL:iH].mean().item(),
            'high': _err[iH:].mean().item(),
        }

    def get_rel_frmse(self):
        _err  = torch.sqrt(self.err_sum  / self.n_total) / self._thw
        _norm = torch.sqrt(self.norm_sum / self.n_total) / self._thw
        rel = _err / _norm
        n = self.n_shells
        iL = int(np.round(n / 3))
        iH = int(np.round(n * 2 / 3))
        return {
            'low':  rel[:iL].mean().item(),
            'mid':  rel[iL:iH].mean().item(),
            'high': rel[iH:].mean().item(),
        }

    def get_psd(self):
        sc = self._smap['shell_count']
        avg = (self.pred_pow_sum / self.n_total).mean(dim=-1)  # [n_shells]
        for s in range(self.n_shells):
            if sc[s] > 0:
                avg[s] /= sc[s]
        return np.arange(self.n_shells), avg.numpy()

    def get_gt_psd(self):
        sc = self._smap['shell_count']
        avg = (self.norm_sum / self.n_total).mean(dim=-1)
        for s in range(self.n_shells):
            if sc[s] > 0:
                avg[s] /= sc[s]
        return np.arange(self.n_shells), avg.numpy()


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(0)
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Parse config ──
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=CONFIG)
    parser.add_argument("--gpu", type=int, default=GPU)
    parser.add_argument("--train_data_type", type=str, default="numerical")
    parser.add_argument("--test_data_type", type=str, default="real")
    parser.add_argument("--checkpoint_path", type=str, default=list(METHODS.values())[0])
    parser.add_argument("--use_hf_dataset", action="store_true", default=True)
    args = parser.parse_args([])
    args = add_args_from_config(args, parser)
    if isinstance(args.gpu, list):
        args.gpu = args.gpu[0]

    # ── Load validation dataset ──
    print("Loading validation dataset...")
    test_dataset = CylinderHFDataset(
        dataset_name='cylinder',
        dataset_root=DATASET_ROOT,
        mode='val',
        N_autoregressive=args.N_autoregressive,
        dataset_type='real',
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=8,
    )
    print(f"Validation dataset: {len(test_dataset)} samples, {len(test_loader)} batches")

    # ── Normalizer ──
    print("Loading normalizer...")
    normalizer_dataset = CylinderHFDataset(
        dataset_name='cylinder',
        dataset_root=DATASET_ROOT,
        mode='train',
        dataset_type='numerical',
    )
    data_normalizer = GaussianNormalizer(normalizer_dataset, device=device)
    train_dataset = normalizer_dataset

    inp0, tgt0 = test_dataset[0]
    T_in, H, W, C_in = inp0.shape
    print(f"Spatial dims: H={H}, W={W}, N_autoregressive={args.N_autoregressive}")

    # ── Per-method inference + spectral accumulation ──
    frmse_results = {}
    psd_results = {}
    gt_psd_done = False
    n_shells = None

    for method_name, ckpt_path in METHODS.items():
        print(f"\n{'='*60}")
        print(f"Inference: {method_name}  [{os.path.basename(ckpt_path)}]")
        print(f"{'='*60}")

        model = load_model(train_dataset, device=device, **vars(args))
        model.load_checkpoint(ckpt_path, device)
        model.eval()

        spec = SpectralAccumulator3D()

        with torch.no_grad():
            for input_data, target_data in tqdm.tqdm(test_loader, desc=method_name):
                unmeasured_c = 0
                for c_ in range(target_data.shape[-1]):
                    if torch.all(target_data[..., c_] == 0):
                        unmeasured_c += 1
                c = target_data.shape[-1] - unmeasured_c

                input_n, target_n = data_normalizer.preprocess(input_data, target_data)

                preds = [input_n]
                for _ in range(args.N_autoregressive):
                    p = model(preds[-1])
                    _, p = data_normalizer.postprocess(preds[-1], p)
                    p, _ = data_normalizer.preprocess(p, target_n)
                    preds.append(p)

                pred = torch.cat(preds[1:], dim=1)
                _, pred = data_normalizer.postprocess(input_n, pred)
                _, target = data_normalizer.postprocess(input_n, target_n)

                pred = pred[..., :c]
                target = target[..., :c]

                pred_cpu = pred.cpu()
                target_cpu = target.cpu()
                del pred, target
                torch.cuda.empty_cache()

                spec.accumulate(pred_cpu, target_cpu)

        frmse_results[method_name] = spec.get_frmse()
        k_bins, psd = spec.get_psd()
        psd_results[method_name] = psd

        if not gt_psd_done:
            _, gt_psd = spec.get_gt_psd()
            psd_results['GT'] = gt_psd
            n_shells = spec.n_shells
            gt_psd_done = True

        fr = frmse_results[method_name]
        print(f"  fRMSE: Low={fr['low']:.6f}  Mid={fr['mid']:.6f}  High={fr['high']:.6f}")

        del model
        torch.cuda.empty_cache()

    # ── Save raw data ──
    iLow = int(np.round(n_shells / 3))
    iHigh = int(np.round(n_shells * 2 / 3))
    raw_data = {
        'frmse': frmse_results,
        'psd_k': k_bins, 'psd': dict(psd_results),
        'data_shape': {'H': H, 'W': W, 'n_val': len(test_dataset)},
        'methods': list(METHODS.keys()),
        'method_colors': METHOD_COLORS,
        'frequency_band_cutoffs': {'n_shells': n_shells, 'iLow': iLow, 'iHigh': iHigh},
    }
    raw_path = os.path.join(OUTPUT_DIR, '004-spectrum_raw_data.pt')
    torch.save(raw_data, raw_path)
    print(f"\nRaw data saved to {raw_path}")

    # ── Generate figure ──
    print("\nGenerating figure...")
    fig, (ax_bar, ax_psd) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.15, wspace=0.30)

    bands = ['Low', 'Mid', 'High']
    band_keys = ['low', 'mid', 'high']
    method_names = list(METHODS.keys())
    n_methods = len(method_names)

    bar_width = 0.14
    x = np.arange(len(bands))

    for i, method in enumerate(method_names):
        vals = [frmse_results[method][bk] for bk in band_keys]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax_bar.bar(x + offset, vals, bar_width,
                   color=METHOD_COLORS[method], label=method,
                   edgecolor='white', linewidth=0.5, zorder=3)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bands, fontsize=11)
    ax_bar.set_ylabel('fRMSE', fontsize=11)
    ax_bar.set_title('(a) Frequency-Band fRMSE', fontsize=12, fontweight='bold', pad=10)
    ax_bar.legend(fontsize=8, ncol=1, loc='upper left', framealpha=0.9)
    ax_bar.grid(axis='y', alpha=0.2, zorder=0)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # Arrow: PhysGuard vs DFT in Low band
    dft_low = frmse_results['DFT']['low']
    pg_low = frmse_results['PhysGuard']['low']
    improvement_low = (dft_low - pg_low) / dft_low * 100

    pg_idx = method_names.index('PhysGuard')
    pg_x = 0 + (pg_idx - (n_methods - 1) / 2) * bar_width
    ax_bar.annotate(
        f'{improvement_low:.1f}% $\\downarrow$',
        xy=(pg_x, pg_low), xytext=(pg_x + 0.30, max(dft_low, pg_low) * 1.35),
        fontsize=9, fontweight='bold', color=METHOD_COLORS['PhysGuard'], ha='center',
        arrowprops=dict(arrowstyle='->', color=METHOD_COLORS['PhysGuard'], lw=1.5),
        zorder=10,
    )

    # (b) PSD
    ax_psd.loglog(k_bins + 1, psd_results['GT'], '-', color='black', lw=2.5,
                  label='GT', zorder=10)

    ls_map = {'Pretrained': '--', 'DFT': '-', 'L2-SP': '-.', 'EWC': ':', 'PhysGuard': '-'}
    lw_map = {'Pretrained': 1.3, 'DFT': 1.5, 'L2-SP': 1.3, 'EWC': 1.3, 'PhysGuard': 2.0}
    for method in method_names:
        ax_psd.loglog(k_bins + 1, psd_results[method],
                      ls_map[method], color=METHOD_COLORS[method],
                      lw=lw_map[method], label=method, alpha=0.85)

    ax_psd.axvspan(1, iLow + 1, alpha=0.10, color='#1E88E5', zorder=0)
    y_lo, y_hi = ax_psd.get_ylim()
    ax_psd.text((iLow + 1) * 0.55, y_hi * 0.35, 'Low-freq\nband',
                fontsize=8, color='#1E88E5', alpha=0.8, ha='center', va='top',
                fontweight='bold')

    ax_psd.set_xlabel('Shell index $k$ (3D)', fontsize=11)
    ax_psd.set_ylabel('Power Spectral Density', fontsize=11)
    ax_psd.set_title('(b) Spherical Shell PSD (3D FFT)', fontsize=12, fontweight='bold', pad=10)
    ax_psd.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax_psd.grid(True, alpha=0.15, which='both')
    ax_psd.spines['top'].set_visible(False)
    ax_psd.spines['right'].set_visible(False)

    fig_path = os.path.join(OUTPUT_DIR, '004-spectrum.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    # ── Summary ──
    print("\n" + "="*70)
    print("FREQUENCY-BAND fRMSE SUMMARY")
    print("="*70)
    print(f"{'Method':12s}  {'Low':>10s}  {'Mid':>10s}  {'High':>10s}")
    print("-"*46)
    for method in method_names:
        fr = frmse_results[method]
        print(f"{method:12s}  {fr['low']:10.6f}  {fr['mid']:10.6f}  {fr['high']:10.6f}")

    for band in band_keys:
        dft_v = frmse_results['DFT'][band]
        pg_v = frmse_results['PhysGuard'][band]
        pct = (dft_v - pg_v) / dft_v * 100
        print(f"PhysGuard vs DFT ({band:4s}): {pct:+.1f}%")


if __name__ == '__main__':
    main()
