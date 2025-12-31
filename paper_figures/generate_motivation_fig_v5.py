"""
Motivation Figure v5 — Real flow field frequency decomposition.

Takes the ACTUAL eval vorticity images from FNO cylinder experiments,
crops out GT and Pred rows, and applies 2D FFT low/mid/high-pass filtering
to show frequency components directly on real results.

Layout:
  4 columns: GT | Pretrained | DFT | PhysGuard
  4 rows:    Full field | Low-freq | Mid-freq | High-freq
  + bottom row: bar chart of fRMSE by frequency band

Usage:
    python scripts/generate_motivation_fig_v5.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import gaussian_filter
import os

# ============================================================================
# Config
# ============================================================================
BASE = './results/001-cylinder/fno'
VORT_IMGS = {
    'GT': None,  # will extract from pretrained's image
    'Pretrained': f'{BASE}/fno_cylinder_pretrained/2026-03-09_17-33-29/eval/figs/sample000_vorticity.png',
    'DFT':        f'{BASE}/fno_cylinder_dft/2026-03-10_03-40-24/eval/figs/sample000_vorticity.png',
    'PhysGuard':  f'{BASE}/fno_cylinder_nsft/2026-03-10_08-42-11/eval/figs/sample000_vorticity.png',
}

# Pixel crop coordinates (from previous analysis):
# Row: GT=133-362, Pred=465-694
# Column (time step 1): 565-1054
CROP_GT = (133, 362)
CROP_PRED = (465, 694)
CROP_COL = (565, 1054)  # 2nd time step

# Colors
C_GT = '#2c3e50'
C_PRE = '#7f8c8d'
C_DFT = '#e74c3c'
C_PG = '#27ae60'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
})


def extract_field(img_path, row='pred', col_range=CROP_COL):
    """Extract a sub-image and convert to grayscale float array."""
    im = np.array(Image.open(img_path))
    r1, r2 = CROP_GT if row == 'gt' else CROP_PRED
    c1, c2 = col_range
    patch = im[r1:r2, c1:c2, :3]  # RGB, drop alpha if present
    # Convert to grayscale (luminance)
    gray = 0.299 * patch[:,:,0] + 0.587 * patch[:,:,1] + 0.114 * patch[:,:,2]
    return gray.astype(np.float64) / 255.0, patch


def freq_decompose(field, low_frac=0.15, mid_frac=0.45):
    """
    2D FFT frequency decomposition into low/mid/high components.
    low_frac: fraction of max frequency for low-pass cutoff
    mid_frac: fraction for mid-pass upper cutoff
    Returns: (low, mid, high) arrays
    """
    H, W = field.shape
    F = np.fft.fft2(field)
    F_shift = np.fft.fftshift(F)
    
    # Create radial frequency grid
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[-cy:H-cy, -cx:W-cx]
    R = np.sqrt(X**2 + Y**2)
    R_max = np.sqrt(cy**2 + cx**2)
    
    # Smooth masks (using Gaussian-like transitions to avoid ringing)
    sigma = R_max * 0.03  # transition width
    
    low_cutoff = R_max * low_frac
    mid_cutoff = R_max * mid_frac
    
    # Low-pass: keep only low frequencies
    mask_low = np.exp(-np.maximum(R - low_cutoff, 0)**2 / (2 * sigma**2))
    mask_low[R <= low_cutoff] = 1.0
    
    # High-pass: only high frequencies
    mask_high = 1.0 - np.exp(-np.maximum(mid_cutoff - R, 0)**2 / (2 * sigma**2))
    mask_high[R >= mid_cutoff] = 1.0
    
    # Mid-pass: everything in between
    mask_mid = 1.0 - mask_low - mask_high
    mask_mid = np.clip(mask_mid, 0, 1)
    # Alternative: explicit band-pass
    mask_mid = np.zeros_like(R, dtype=float)
    mask_mid[(R > low_cutoff) & (R <= mid_cutoff)] = 1.0
    # Smooth edges
    transition = (R > low_cutoff * 0.8) & (R <= low_cutoff * 1.2)
    mask_mid[transition] = np.clip((R[transition] - low_cutoff * 0.8) / (low_cutoff * 0.4), 0, 1)
    transition2 = (R > mid_cutoff * 0.85) & (R <= mid_cutoff * 1.15)
    mask_mid[transition2] *= np.clip(1 - (R[transition2] - mid_cutoff * 0.85) / (mid_cutoff * 0.3), 0, 1)
    
    low = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift * mask_low)))
    mid = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift * mask_mid)))
    high = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift * mask_high)))
    
    return low, mid, high


def main():
    # ---- Extract fields ----
    methods = ['GT', 'Pretrained', 'DFT', 'PhysGuard']
    colors = [C_GT, C_PRE, C_DFT, C_PG]
    
    fields_gray = {}
    fields_rgb = {}
    
    # GT comes from pretrained image's GT row
    gt_gray, gt_rgb = extract_field(VORT_IMGS['Pretrained'], row='gt')
    fields_gray['GT'] = gt_gray
    fields_rgb['GT'] = gt_rgb
    
    for m in ['Pretrained', 'DFT', 'PhysGuard']:
        gray, rgb = extract_field(VORT_IMGS[m], row='pred')
        fields_gray[m] = gray
        fields_rgb[m] = rgb
    
    # ---- Frequency decomposition ----
    decomposed = {}
    for m in methods:
        low, mid, high = freq_decompose(fields_gray[m])
        decomposed[m] = {'low': low, 'mid': mid, 'high': high}
    
    # ---- Figure layout ----
    # 5 rows: Full RGB | Low-freq | Mid-freq | High-freq | Error heatmap
    # 4 cols: GT | Pretrained | DFT | PhysGuard
    fig = plt.figure(figsize=(15, 14))
    gs = GridSpec(5, 4, figure=fig, hspace=0.15, wspace=0.05,
                  left=0.08, right=0.92, top=0.95, bottom=0.06,
                  height_ratios=[1, 1, 1, 1, 0.65])
    
    row_labels = ['Full Field\n(original)', 'Low-Freq\n(large-scale\nstructure)',
                  'Mid-Freq\n(medium\nscales)', 'High-Freq\n(fine details\n& noise)']
    
    # ---- Rows 0-3: fields ----
    for r in range(4):
        for c, m in enumerate(methods):
            ax = fig.add_subplot(gs[r, c])
            
            if r == 0:
                # Full field: show RGB
                ax.imshow(fields_rgb[m])
                ax.set_title(m, fontsize=13, fontweight='bold', color=colors[c], pad=8)
            else:
                # Frequency components: show as heatmap
                comp_keys = ['low', 'mid', 'high']
                data = decomposed[m][comp_keys[r-1]]
                
                # Use consistent colormap range
                if r == 1:  # low-freq
                    vmin, vmax = 0.15, 0.85
                    cmap = 'RdBu_r'
                elif r == 2:  # mid-freq
                    vmin, vmax = -0.12, 0.12
                    cmap = 'RdBu_r'
                else:  # high-freq
                    vmin, vmax = -0.06, 0.06
                    cmap = 'RdBu_r'
                
                ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Row labels on the left
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10, fontweight='bold',
                              rotation=0, ha='right', va='center', labelpad=50,
                              color='#333')
            
            # Highlight DFT low-freq degradation
            if m == 'DFT' and r == 1:
                for spine in ax.spines.values():
                    spine.set_linewidth(3)
                    spine.set_color(C_DFT)
                ax.text(0.5, 0.02, 'LOW-FREQ DEGRADED',
                        transform=ax.transAxes, fontsize=9, fontweight='bold',
                        color='white', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_DFT,
                                  alpha=0.85, edgecolor='none'))
            
            # Highlight PhysGuard low-freq preserved
            if m == 'PhysGuard' and r == 1:
                for spine in ax.spines.values():
                    spine.set_linewidth(3)
                    spine.set_color(C_PG)
                ax.text(0.5, 0.02, 'LOW-FREQ PRESERVED',
                        transform=ax.transAxes, fontsize=9, fontweight='bold',
                        color='white', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_PG,
                                  alpha=0.85, edgecolor='none'))
            
            # Highlight Pretrained high-freq blurry
            if m == 'Pretrained' and r == 3:
                for spine in ax.spines.values():
                    spine.set_linewidth(2.5)
                    spine.set_color('#e67e22')
                ax.text(0.5, 0.02, 'BLURRY',
                        transform=ax.transAxes, fontsize=9, fontweight='bold',
                        color='white', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e67e22',
                                  alpha=0.85, edgecolor='none'))
            
            # Normal borders
            if not (m == 'DFT' and r == 1) and not (m == 'PhysGuard' and r == 1) \
                    and not (m == 'Pretrained' and r == 3):
                for spine in ax.spines.values():
                    spine.set_linewidth(1)
                    spine.set_color('#ccc')
    
    # ---- Row 4: Bar chart comparison ----
    # Using actual FNO cylinder fRMSE values
    freq_data = {
        'Pretrained': {'Low': 0.01829, 'Mid': 0.01344, 'High': 0.00600},
        'DFT':        {'Low': 0.01651, 'Mid': 0.00999, 'High': 0.00512},
        'PhysGuard':  {'Low': 0.01131, 'Mid': 0.01042, 'High': 0.00528},
    }
    
    bands = ['Low-freq\nfRMSE', 'Mid-freq\nfRMSE', 'High-freq\nfRMSE']
    method_keys = ['Pretrained', 'DFT', 'PhysGuard']
    bar_colors = [C_PRE, C_DFT, C_PG]
    band_keys = ['Low', 'Mid', 'High']
    
    ax_bar = fig.add_subplot(gs[4, :])
    
    x = np.arange(len(bands))
    width = 0.22
    offsets = [-width, 0, width]
    
    for i, (mk, clr) in enumerate(zip(method_keys, bar_colors)):
        vals = [freq_data[mk][bk] for bk in band_keys]
        bars = ax_bar.bar(x + offsets[i], vals, width, color=clr, edgecolor='white',
                          linewidth=1.5, label=mk, zorder=3)
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=clr)
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bands, fontsize=11, fontweight='bold')
    ax_bar.set_ylabel('fRMSE  $\\downarrow$', fontsize=12, fontweight='bold')
    ax_bar.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax_bar.grid(axis='y', alpha=0.2)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_ylim(0, 0.025)
    
    # Annotate DFT low-freq is WORST among the three despite best total RMSE
    # Arrow pointing to DFT's low-freq bar
    ax_bar.annotate('DFT: best total RMSE,\nbut worst low-freq fRMSE!',
                    xy=(0 + 0, freq_data['DFT']['Low']),
                    xytext=(0.8, 0.022),
                    fontsize=10, color=C_DFT, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_DFT, lw=2),
                    ha='center', va='top')
    
    # PhysGuard best low-freq
    ax_bar.annotate('PhysGuard:\nbest low-freq!',
                    xy=(0 + width, freq_data['PhysGuard']['Low']),
                    xytext=(0.8, 0.005),
                    fontsize=10, color=C_PG, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_PG, lw=2),
                    ha='center', va='bottom')
    
    # ---- Save ----
    out_dir = "./imgs"
    os.makedirs(out_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        path = os.path.join(out_dir, f'figure1_motivation_v5.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
