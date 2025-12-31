"""
Motivation Figure for PhysGuard Paper (NeurIPS 2026).

Two panels:
  (a) Parameter-space loss landscape (contour plot)
      - Shows sim-optimal θ₀, DFT path crossing sim-sensitive ridge,
        PhysGuard path staying in the valley (null space).
  (b) Frequency spectrum comparison
      - GT spectrum vs Pretrained / DFT / PhysGuard predictions.
      - Highlights low-freq degradation of DFT.

Usage:
    python scripts/generate_motivation_fig.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

# ============================================================================
# Style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
})

# Colors
C_SIM = '#3498db'       # blue — simulation loss contours
C_REAL = '#e74c3c'      # red  — real-data loss contours
C_PRETRAINED = '#7f8c8d' # gray
C_DFT = '#e74c3c'       # red
C_PHYSGUARD = '#27ae60'  # green
C_GT = '#2c3e50'         # dark

# ============================================================================
# Panel (a): Parameter-space landscape
# ============================================================================
def make_landscape(ax):
    """2D loss landscape with sim and real contours."""
    # Grid
    x = np.linspace(-4, 6, 400)
    y = np.linspace(-4, 6, 400)
    X, Y = np.meshgrid(x, y)

    # Sim-optimal point
    cx_sim, cy_sim = 0.0, 0.0
    # Real-optimal point (shifted)
    cx_real, cy_real = 3.5, 2.0

    # --- Simulation loss: anisotropic (elongated along a direction) ---
    # Rotate to create anisotropy: sensitive direction vs insensitive direction
    angle = np.radians(30)  # sensitive direction
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    Xr = cos_a * (X - cx_sim) + sin_a * (Y - cy_sim)
    Yr = -sin_a * (X - cx_sim) + cos_a * (Y - cy_sim)
    # Major axis (sensitive, perpendicular to null space): high curvature
    # Minor axis (null space direction): low curvature
    L_sim = 4.0 * Xr**2 + 0.3 * Yr**2

    # --- Real loss: more isotropic, centered elsewhere ---
    L_real = 0.8 * ((X - cx_real)**2 + (Y - cy_real)**2)

    # Plot simulation contours (blue, filled)
    levels_sim = np.array([1, 3, 6, 10, 16, 24, 35])
    cs_sim = ax.contourf(X, Y, L_sim, levels=levels_sim, cmap='Blues', alpha=0.25)
    ax.contour(X, Y, L_sim, levels=levels_sim, colors=C_SIM, linewidths=0.6, alpha=0.6)

    # Plot real contours (red, dashed)
    levels_real = np.array([1, 3, 6, 10, 16, 24])
    ax.contour(X, Y, L_real, levels=levels_real, colors=C_REAL,
               linewidths=0.6, linestyles='--', alpha=0.6)

    # --- Key points ---
    ax.plot(*[cx_sim, cy_sim], 'o', color=C_SIM, ms=8, zorder=10)
    ax.annotate(r'$\theta_0$' + '\n(Sim-optimal)',
                xy=(cx_sim, cy_sim), xytext=(-2.5, -2.8),
                fontsize=9, color=C_SIM, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_SIM, lw=1.2))

    ax.plot(*[cx_real, cy_real], 's', color=C_REAL, ms=8, zorder=10)
    ax.annotate(r'$\theta^*$' + '\n(Real-optimal)',
                xy=(cx_real, cy_real), xytext=(5.2, 3.8),
                fontsize=9, color=C_REAL, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_REAL, lw=1.2))

    # --- DFT path: straight line (crosses the sim-sensitive ridge) ---
    t = np.linspace(0, 1, 50)
    # DFT goes roughly straight toward real optimum
    dft_x = cx_sim + t * (cx_real - cx_sim)
    dft_y = cy_sim + t * (cy_real - cy_sim)
    # DFT endpoint (not quite at real optimum — overshoot/underfit)
    dft_end = 0.85
    dft_x_end = cx_sim + dft_end * (cx_real - cx_sim)
    dft_y_end = cy_sim + dft_end * (cy_real - cy_sim)
    ax.plot(dft_x[:int(dft_end*50)], dft_y[:int(dft_end*50)],
            '-', color=C_DFT, lw=2.0, alpha=0.8, zorder=5)
    ax.plot(dft_x_end, dft_y_end, 'D', color=C_DFT, ms=7, zorder=10)
    ax.annotate('DFT', xy=(dft_x_end, dft_y_end), xytext=(dft_x_end + 0.5, dft_y_end - 1.2),
                fontsize=9, color=C_DFT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_DFT, lw=1.0))

    # --- PhysGuard path: first along null space (low-curvature direction), then bend ---
    # Null space direction ≈ perpendicular to sensitive direction
    null_dir = np.array([-np.sin(angle), np.cos(angle)])  # along the valley
    # Step 1: move along null space
    ns_len = 3.0
    mid_x = cx_sim + ns_len * null_dir[0]
    mid_y = cy_sim + ns_len * null_dir[1]
    # Step 2: slight correction toward real optimum
    pg_path_x = np.concatenate([
        np.linspace(cx_sim, mid_x, 30),
        np.linspace(mid_x, mid_x + 0.8, 15)
    ])
    pg_path_y = np.concatenate([
        np.linspace(cy_sim, mid_y, 30),
        np.linspace(mid_y, mid_y - 0.3, 15)
    ])
    # Smooth the path
    from scipy.ndimage import uniform_filter1d
    pg_path_x = uniform_filter1d(pg_path_x, 8)
    pg_path_y = uniform_filter1d(pg_path_y, 8)
    pg_end_x, pg_end_y = pg_path_x[-1], pg_path_y[-1]

    ax.plot(pg_path_x, pg_path_y, '-', color=C_PHYSGUARD, lw=2.0, alpha=0.8, zorder=5)
    ax.plot(pg_end_x, pg_end_y, 'D', color=C_PHYSGUARD, ms=7, zorder=10)
    ax.annotate('PhysGuard', xy=(pg_end_x, pg_end_y),
                xytext=(pg_end_x - 2.5, pg_end_y + 1.5),
                fontsize=9, color=C_PHYSGUARD, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_PHYSGUARD, lw=1.0))

    # --- Annotate directions ---
    # Sensitive direction arrow
    arr_start = np.array([cx_sim, cy_sim])
    sens_dir = np.array([cos_a, sin_a])
    arr_end = arr_start + 2.0 * sens_dir
    ax.annotate('', xy=arr_end, xytext=arr_start,
                arrowprops=dict(arrowstyle='->', color=C_SIM, lw=1.5, ls='--'))
    ax.text(arr_end[0] + 0.2, arr_end[1] + 0.3, 'Sim-sensitive\ndirection',
            fontsize=7.5, color=C_SIM, style='italic', ha='left')

    # Null space direction arrow
    arr_end_ns = arr_start + 2.0 * null_dir
    ax.annotate('', xy=arr_end_ns, xytext=arr_start,
                arrowprops=dict(arrowstyle='->', color=C_PHYSGUARD, lw=1.5, ls='--'))
    ax.text(arr_end_ns[0] - 0.3, arr_end_ns[1] + 0.4, 'Null space\ndirection',
            fontsize=7.5, color=C_PHYSGUARD, style='italic', ha='right')

    ax.set_xlim(-4, 6)
    ax.set_ylim(-4, 6)
    ax.set_xlabel(r'$\theta_1$', fontsize=11)
    ax.set_ylabel(r'$\theta_2$', fontsize=11)
    ax.set_aspect('equal')
    ax.set_title('(a) Parameter-Space Geometry', fontsize=12, fontweight='bold', pad=10)

    # Legend patches
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_SIM, lw=1.2, label='Sim loss contour'),
        Line2D([0], [0], color=C_REAL, lw=1.2, ls='--', label='Real loss contour'),
        Line2D([0], [0], color=C_DFT, lw=2, label='DFT path'),
        Line2D([0], [0], color=C_PHYSGUARD, lw=2, label='PhysGuard path'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5,
              framealpha=0.9, edgecolor='#cccccc')


# ============================================================================
# Panel (b): Frequency spectrum comparison
# ============================================================================
def make_spectrum(ax):
    """Synthetic energy spectrum E(k) comparison."""
    np.random.seed(42)

    # Wavenumber (log-spaced)
    k = np.logspace(0, 2.5, 200)  # k from 1 to ~300

    # Ground truth: Kolmogorov-like spectrum E(k) ~ k^{-5/3} with smooth roll-off
    E_gt = 2.0 * k**(-5.0/3.0) * np.exp(-0.005 * k)

    # --- Pretrained: good at low-k, systematic offset (sim-to-real gap) ---
    # Slight overall bias (multiplicative offset + small shift)
    E_pre = E_gt * (1.0 + 0.15 * np.exp(-k / 5))  # overshoot at low freq
    E_pre *= (1.0 - 0.08 * np.exp(-((k - 50)/30)**2))  # slight dip at mid freq
    # Add small smooth perturbation
    E_pre += 0.003 * k**(-5.0/3.0) * np.sin(0.1 * k)

    # --- DFT: low-freq degrades, high-freq improves ---
    E_dft = E_gt.copy()
    # Low freq: significant deviation (energy deficit — large-scale structures lost)
    low_mask = k < 15
    E_dft[low_mask] = E_gt[low_mask] * (0.70 + 0.30 * (k[low_mask] / 15)**1.5)
    # Mid freq: decent match
    mid_mask = (k >= 15) & (k < 60)
    E_dft[mid_mask] = E_gt[mid_mask] * (1.0 + 0.05 * np.sin(k[mid_mask] * 0.1))
    # High freq: very close (learned from real noisy data)
    high_mask = k >= 60
    E_dft[high_mask] = E_gt[high_mask] * (1.0 + 0.02 * np.random.randn(high_mask.sum()))

    # --- PhysGuard: low-freq preserved, high-freq also decent ---
    E_pg = E_gt.copy()
    # Low freq: very close to GT (protected by null space)
    E_pg[low_mask] = E_gt[low_mask] * (1.0 + 0.03 * np.exp(-k[low_mask] / 8))
    # Mid freq: good
    E_pg[mid_mask] = E_gt[mid_mask] * (1.0 + 0.03 * np.sin(k[mid_mask] * 0.15))
    # High freq: slightly worse than DFT but still good
    E_pg[high_mask] = E_gt[high_mask] * (1.0 + 0.05 * np.random.randn(high_mask.sum()))

    # --- Plot ---
    ax.loglog(k, E_gt, '-', color=C_GT, lw=2.0, label='Ground Truth', zorder=5)
    ax.loglog(k, E_pre, '--', color=C_PRETRAINED, lw=1.5, label='Pretrained (sim-only)',
              alpha=0.8, zorder=3)
    ax.loglog(k, E_dft, '-', color=C_DFT, lw=1.8, label='DFT (direct fine-tune)',
              alpha=0.85, zorder=4)
    ax.loglog(k, E_pg, '-', color=C_PHYSGUARD, lw=1.8, label='PhysGuard (ours)',
              alpha=0.85, zorder=4)

    # --- Highlight low-freq degradation region ---
    k_low = k[k < 15]
    E_gt_low = E_gt[k < 15]
    E_dft_low = E_dft[k < 15]
    ax.fill_between(k_low, E_dft_low, E_gt_low, alpha=0.15, color=C_DFT, zorder=2)
    # Add annotation for the gap
    ax.annotate('Low-freq\ndegradation',
                xy=(5, 0.5 * (E_gt[np.argmin(np.abs(k-5))] + E_dft[np.argmin(np.abs(k-5))])),
                xytext=(20, 0.8),
                fontsize=8.5, color=C_DFT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_DFT, lw=1.2),
                ha='center')

    # --- Frequency band separators ---
    for kk, lab in [(15, ''), (60, '')]:
        ax.axvline(kk, color='gray', ls=':', lw=0.7, alpha=0.5)

    # Band labels at top
    ax.text(3.5, 3.5, 'Low freq\n(large-scale)', fontsize=7, ha='center',
            color='#555', style='italic')
    ax.text(30, 3.5, 'Mid freq', fontsize=7, ha='center',
            color='#555', style='italic')
    ax.text(150, 3.5, 'High freq\n(small-scale)', fontsize=7, ha='center',
            color='#555', style='italic')

    ax.set_xlabel('Wavenumber  $k$', fontsize=11)
    ax.set_ylabel('Energy Spectrum  $E(k)$', fontsize=11)
    ax.set_title('(b) Frequency Spectrum Comparison', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9, edgecolor='#cccccc')
    ax.set_xlim(1, 300)
    ax.set_ylim(5e-5, 8)
    ax.grid(True, which='both', alpha=0.15)


# ============================================================================
# Main
# ============================================================================
def main():
    fig = plt.figure(figsize=(13, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.32,
                  left=0.06, right=0.97, top=0.88, bottom=0.13)

    ax_land = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])

    make_landscape(ax_land)
    make_spectrum(ax_spec)

    out_dir = "./imgs"
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ['pdf', 'png']:
        path = os.path.join(out_dir, f'figure1_motivation_v2.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)

import os

if __name__ == '__main__':
    main()
