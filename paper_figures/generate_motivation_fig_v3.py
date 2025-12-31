"""
Motivation Figure v3 — visually striking, NeurIPS-quality.

Panel (a): 3D loss landscape surface
   - Dramatic mountain/valley terrain
   - DFT path climbs over the sim-sensitive ridge (red)
   - PhysGuard path follows the valley floor (green)
   - Clear camera angle + lighting

Panel (b): Synthetic 1D signal decomposition
   - Top row: full signal
   - Middle row: low-freq component (large-scale structure)
   - Bottom row: high-freq component (fine details)
   - 4 columns: GT / Pretrained / DFT / PhysGuard
   - Immediately shows "DFT lost the low-freq shape"

Usage:
    cd .
    python scripts/generate_motivation_fig_v3.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
import os

# ============================================================================
# Style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
})

C_DFT = '#e74c3c'
C_PG = '#27ae60'
C_GT = '#2c3e50'
C_PRE = '#7f8c8d'

# ============================================================================
# Panel (a): 3D Loss Landscape
# ============================================================================
def make_3d_landscape(ax):
    """3D surface plot with DFT vs PhysGuard paths."""

    # --- Build terrain ---
    N = 300
    x = np.linspace(-3, 5, N)
    y = np.linspace(-3, 5, N)
    X, Y = np.meshgrid(x, y)

    # Sim loss: anisotropic bowl rotated 30°
    angle = np.radians(30)
    ca, sa = np.cos(angle), np.sin(angle)
    Xr = ca * X + sa * Y
    Yr = -sa * X + ca * Y
    L_sim = 3.5 * Xr**2 + 0.25 * Yr**2  # steep in sensitive dir, flat in null space dir

    # Add a ridge along the sensitive direction between sim and real optima
    ridge_center_x, ridge_center_y = 1.5, 1.0
    ridge = 6.0 * np.exp(-0.8 * ((X - ridge_center_x)**2 / 0.8 + (Y - ridge_center_y)**2 / 3.0))
    # Rotate the ridge
    Xrr = ca * (X - ridge_center_x) + sa * (Y - ridge_center_y)
    Yrr = -sa * (X - ridge_center_x) + ca * (Y - ridge_center_y)
    ridge = 5.0 * np.exp(-2.0 * Xrr**2 - 0.15 * Yrr**2)

    # Real loss: bowl centered elsewhere
    cx_real, cy_real = 3.0, 1.5
    L_real = 1.2 * ((X - cx_real)**2 + (Y - cy_real)**2)

    # Combined landscape: sim + ridge + weak real influence
    Z = 0.6 * L_sim + ridge + 0.08 * L_real
    # Normalize
    Z = Z / Z.max() * 12
    # Smooth
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, sigma=3)

    # --- Surface ---
    # Custom colormap: blues for valleys, warm for ridges
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm',
                           alpha=0.75, linewidth=0, antialiased=True,
                           rstride=3, cstride=3, zorder=1)

    # --- Paths ---
    # Start point (sim-optimal)
    sx, sy = 0.0, 0.0

    # DFT path: goes roughly straight toward real optimum → crosses the ridge
    t_dft = np.linspace(0, 1, 100)
    dft_x = sx + t_dft * 2.8
    dft_y = sy + t_dft * 1.3
    # Get Z values on the surface
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((y, x), Z, method='linear', bounds_error=False, fill_value=None)
    dft_z = np.array([interp([dy, dx])[0] for dx, dy in zip(dft_x, dft_y)])
    dft_z += 0.15  # lift path slightly above surface
    ax.plot(dft_x, dft_y, dft_z, '-', color=C_DFT, lw=3.5, zorder=10, label='DFT')
    ax.scatter([dft_x[-1]], [dft_y[-1]], [dft_z[-1]], color=C_DFT, s=80,
               marker='D', zorder=11, edgecolors='white', linewidths=1.0)

    # PhysGuard path: follows the valley (null space direction), then curves toward real
    null_dir = np.array([-sa, ca])  # perpendicular to sensitive direction
    t_pg = np.linspace(0, 1, 100)
    # First ~70%: along null space; last 30%: gentle curve toward real
    pg_x = np.zeros(100)
    pg_y = np.zeros(100)
    for i, ti in enumerate(t_pg):
        if ti < 0.7:
            s = ti / 0.7
            pg_x[i] = sx + s * 2.5 * null_dir[0]
            pg_y[i] = sy + s * 2.5 * null_dir[1]
        else:
            s = (ti - 0.7) / 0.3
            base_x = sx + 2.5 * null_dir[0]
            base_y = sy + 2.5 * null_dir[1]
            tgt_x = 2.2
            tgt_y = 2.0
            pg_x[i] = base_x + s * (tgt_x - base_x)
            pg_y[i] = base_y + s * (tgt_y - base_y)

    # Smooth path
    pg_x = gaussian_filter1d(pg_x, 5)
    pg_y = gaussian_filter1d(pg_y, 5)
    pg_z = np.array([interp([py, px])[0] for px, py in zip(pg_x, pg_y)])
    pg_z += 0.15
    ax.plot(pg_x, pg_y, pg_z, '-', color=C_PG, lw=3.5, zorder=10, label='PhysGuard')
    ax.scatter([pg_x[-1]], [pg_y[-1]], [pg_z[-1]], color=C_PG, s=80,
               marker='D', zorder=11, edgecolors='white', linewidths=1.0)

    # Start point
    sz = interp([sy, sx])[0] + 0.15
    ax.scatter([sx], [sy], [sz], color='#3498db', s=120, marker='o', zorder=11,
               edgecolors='white', linewidths=1.5)
    ax.text(sx - 0.5, sy - 0.8, sz + 1.0, r'$\theta_0$' + '\n(Pretrained)',
            fontsize=10, color='#3498db', fontweight='bold', ha='center', zorder=12)

    # End labels
    ax.text(dft_x[-1] + 0.3, dft_y[-1] - 0.3, dft_z[-1] + 0.8, 'DFT',
            fontsize=10, color=C_DFT, fontweight='bold', ha='center', zorder=12)
    ax.text(pg_x[-1] - 0.7, pg_y[-1] + 0.3, pg_z[-1] + 0.8, 'PhysGuard',
            fontsize=10, color=C_PG, fontweight='bold', ha='center', zorder=12)

    # Ridge annotation
    ridge_peak_z = interp([ridge_center_y, ridge_center_x])[0]
    ax.text(ridge_center_x + 0.2, ridge_center_y - 1.5, ridge_peak_z + 1.5,
            'Sim-sensitive\n   ridge',
            fontsize=8.5, color='#8e44ad', style='italic', ha='center', zorder=12)

    # Valley annotation
    valley_x = sx + 1.2 * null_dir[0]
    valley_y = sy + 1.2 * null_dir[1]
    valley_z = interp([valley_y, valley_x])[0]
    ax.text(valley_x - 1.0, valley_y + 0.3, valley_z + 0.6,
            'Null space\n  valley',
            fontsize=8.5, color=C_PG, style='italic', ha='center', zorder=12)

    # --- Camera & formatting ---
    ax.view_init(elev=32, azim=-55)
    ax.set_xlabel(r'$\theta_1$', fontsize=11, labelpad=5)
    ax.set_ylabel(r'$\theta_2$', fontsize=11, labelpad=5)
    ax.set_zlabel('Loss', fontsize=11, labelpad=3)
    ax.set_title('(a) Parameter-Space Landscape', fontsize=13, fontweight='bold',
                 pad=12, loc='center')

    # Clean up
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#e0e0e0')
    ax.yaxis.pane.set_edgecolor('#e0e0e0')
    ax.zaxis.pane.set_edgecolor('#e0e0e0')
    ax.grid(True, alpha=0.15)


# ============================================================================
# Panel (b): Synthetic signal decomposition — 1D "flow field"
# ============================================================================
def make_signal_decomposition(fig, gs_right):
    """
    3 rows × 4 columns of 1D signals showing:
      Row 1: Full signal
      Row 2: Low-freq component (large-scale structure)
      Row 3: High-freq component (small-scale details)
      Cols: GT / Pretrained / DFT / PhysGuard
    """
    np.random.seed(42)

    # --- Generate synthetic 1D "flow field" ---
    N = 500
    x = np.linspace(0, 2 * np.pi, N)

    # Low-freq component (large-scale physics): dominant vortex shedding
    low_gt = 1.8 * np.sin(2 * x) + 0.9 * np.sin(3 * x + 0.5) + 0.5 * np.cos(x + 0.3)

    # High-freq component (small-scale details): turbulent fluctuations
    high_gt = (0.3 * np.sin(15 * x) + 0.2 * np.sin(22 * x + 1.0) +
               0.15 * np.sin(35 * x + 0.5) + 0.1 * np.random.randn(N) * 0.5)
    high_gt = gaussian_filter1d(high_gt, 1.5)

    full_gt = low_gt + high_gt

    # --- Pretrained: good low-freq, systematic bias ---
    low_pre = low_gt * 1.12 + 0.25  # slight amplitude overshoot + offset (sim-real gap)
    high_pre = high_gt * 0.5 + 0.08 * np.sin(18 * x)  # blurry high-freq
    full_pre = low_pre + high_pre

    # --- DFT: low-freq degrades, high-freq matches well ---
    low_dft = low_gt * 0.72 + 0.15 * np.sin(2.5 * x + 1.0)  # amplitude shrinks, phase shifts
    high_dft = high_gt * 0.92 + 0.05 * np.random.randn(N)  # good high-freq
    high_dft = gaussian_filter1d(high_dft, 1.0)
    full_dft = low_dft + high_dft

    # --- PhysGuard: both good ---
    low_pg = low_gt * 1.03 + 0.05  # almost perfect low-freq
    high_pg = high_gt * 0.85 + 0.04 * np.random.randn(N)  # decent high-freq
    high_pg = gaussian_filter1d(high_pg, 1.2)
    full_pg = low_pg + high_pg

    # --- Data dict ---
    data = {
        'GT':         {'full': full_gt,  'low': low_gt,  'high': high_gt},
        'Pretrained': {'full': full_pre, 'low': low_pre, 'high': high_pre},
        'DFT':        {'full': full_dft, 'low': low_dft, 'high': high_dft},
        'PhysGuard':  {'full': full_pg,  'low': low_pg,  'high': high_pg},
    }
    col_labels = ['Ground Truth', 'Pretrained\n(sim-only)', 'DFT\n(fine-tuned)', 'PhysGuard\n(ours)']
    col_keys = ['GT', 'Pretrained', 'DFT', 'PhysGuard']
    row_labels = ['Full Signal', 'Low-Freq\n(large-scale)', 'High-Freq\n(small-scale)']
    row_keys = ['full', 'low', 'high']
    col_colors = [C_GT, C_PRE, C_DFT, C_PG]

    inner_gs = gs_right.subgridspec(3, 4, hspace=0.35, wspace=0.12)

    for r, rk in enumerate(row_keys):
        for c, ck in enumerate(col_keys):
            ax = fig.add_subplot(inner_gs[r, c])
            signal = data[ck][rk]
            gt_signal = data['GT'][rk]

            # Fill between GT and prediction
            if ck != 'GT':
                ax.fill_between(x, gt_signal, signal, alpha=0.15, color=col_colors[c])
                ax.plot(x, gt_signal, '-', color=C_GT, lw=0.6, alpha=0.4, zorder=2)

            ax.plot(x, signal, '-', color=col_colors[c], lw=1.3, zorder=3)

            # Highlight problematic region for DFT low-freq
            if ck == 'DFT' and rk == 'low':
                ax.fill_between(x, gt_signal, signal, alpha=0.30, color=C_DFT, zorder=2)
                # Big red X or warning
                mid_idx = N // 2
                ax.annotate('degraded!', xy=(x[mid_idx], signal[mid_idx]),
                            xytext=(x[mid_idx] + 0.8, signal[mid_idx] + 1.0),
                            fontsize=7.5, color=C_DFT, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=C_DFT, lw=1.0))

            # Checkmark for PhysGuard low-freq
            if ck == 'PhysGuard' and rk == 'low':
                ax.text(0.92, 0.85, 'preserved', transform=ax.transAxes,
                        fontsize=7, color=C_PG, fontweight='bold', ha='right',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#d5f5e3',
                                  edgecolor=C_PG, alpha=0.8))

            # Column titles
            if r == 0:
                title_color = col_colors[c]
                ax.set_title(col_labels[c], fontsize=9, fontweight='bold',
                             color=title_color, pad=6)

            # Row labels
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=8.5, fontweight='bold',
                              color='#333', rotation=0, ha='right', va='center',
                              labelpad=45)

            # Clean up axes
            ax.set_xlim(x[0], x[-1])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('#cccccc')

            # Y-range per row
            if rk == 'full':
                ax.set_ylim(-3.5, 4.5)
            elif rk == 'low':
                ax.set_ylim(-3.5, 4.0)
            else:
                ax.set_ylim(-1.2, 1.2)

    # Panel title
    fig.text(0.72, 0.96, '(b) Signal Frequency Decomposition',
             fontsize=13, fontweight='bold', ha='center', va='top')


# ============================================================================
# Main
# ============================================================================
def main():
    fig = plt.figure(figsize=(16, 7.5))

    # Left: 3D landscape (wider), Right: signal decomposition grid
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4],
                  wspace=0.08, left=0.02, right=0.98, top=0.92, bottom=0.04)

    # Panel (a): 3D
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    make_3d_landscape(ax3d)

    # Panel (b): Signal decomposition
    make_signal_decomposition(fig, gs[0, 1])

    out_dir = "./imgs"
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ['pdf', 'png']:
        path = os.path.join(out_dir, f'figure1_motivation_v3.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {path}')

    plt.close(fig)


if __name__ == '__main__':
    main()
