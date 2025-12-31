"""
Generate Figure 1 (Motivation Figure) for the PhysGuard paper.

Layout:  (a) Conceptual diagram  |  (b) Flow field comparison  |  (c) Frequency bar chart

Panel (b) uses precise pixel cropping from existing eval vorticity images.

Usage:
    cd ./RealPDEBench
    conda activate pytorch310
    python ./scripts/generate_figure1.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================
WORKSPACE = "."
RESULTS_DIR = os.path.join(WORKSPACE, "RealPDEBench/results/001-cylinder/fno")
OUTPUT_DIR = os.path.join(WORKSPACE, "RealPDEBench/imgs")

# Existing vorticity images (FNO, cylinder, sample 000)
VORT_IMGS = {
    'Pretrained': os.path.join(RESULTS_DIR, "fno_cylinder_pretrained/2026-03-09_17-33-29/eval/figs/sample000_vorticity.png"),
    'DFT':        os.path.join(RESULTS_DIR, "fno_cylinder_dft/2026-03-10_03-40-24/eval/figs/sample000_vorticity.png"),
    'PhysGuard':  os.path.join(RESULTS_DIR, "fno_cylinder_nsft/2026-03-10_08-42-11/eval/figs/sample000_vorticity.png"),
}

# Pixel coordinates for cropping (empirically determined from image analysis)
# Vorticity images are 2148×1054 with 3 rows × 4 columns of subplots
# Row boundaries: GT=133-362, Pred=465-694, Error=793-1016
# Column boundaries (plot area only, 4 time steps):
#   Col0=36-442, Col1=565-971, Col2=1095-1501, Col3=1624-2030
CROP_ROWS = {'gt': (133, 362), 'pred': (465, 694), 'error': (793, 1016)}
CROP_COLS = [(36, 525), (565, 1054), (1095, 1584), (1624, 2113)]  # include colorbar
COL_IDX = 1  # Use 2nd time step

FREQ_DATA = {
    'FNO': {
        'Pretrained': [0.01829, 0.01344, 0.006],
        'DFT':        [0.01651, 0.00999, 0.00512],
        'EWC-FT':     [0.01596, 0.00994, 0.00514],
        'PhysGuard':  [0.01131, 0.01042, 0.00528],
    },
    'CNO': {
        'Pretrained': [0.01379, 0.01296, 0.00585],
        'DFT':        [0.01585, 0.00650, 0.00433],
        'EWC-FT':     [0.01355, 0.00605, 0.00414],
        'PhysGuard':  [0.01148, 0.00705, 0.00443],
    },
    'DPOT': {
        'Pretrained': [0.01202, 0.00841, 0.00471],
        'DFT':        [0.01026, 0.00620, 0.00372],
        'EWC-FT':     [0.00989, 0.00629, 0.00373],
        'PhysGuard':  [0.00998, 0.00644, 0.00386],
    },
    'DeepONet': {
        'Pretrained': [0.02793, 0.01486, 0.006],
        'DFT':        [0.02527, 0.01421, 0.00601],
        'EWC-FT':     [0.02599, 0.01174, 0.00585],
        'PhysGuard':  [0.01957, 0.01207, 0.00585],
    },
}

# ============================================================================
# Style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
})

C_PRETRAINED = '#888888'
C_DFT = '#e74c3c'
C_EWC = '#f39c12'
C_NSFT = '#2ecc71'
C_PHYSICS = '#3498db'


# ============================================================================
# Panel (a): Conceptual Parameter-Space Diagram
# ============================================================================
def draw_concept_diagram(ax):
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-2.0, 3.0)
    ax.set_aspect('equal')
    ax.axis('off')

    theta_star = np.array([0.0, 0.5])
    phi = np.radians(25)
    phys_dir = np.array([np.cos(phi), np.sin(phi)])
    phys_normal = np.array([-np.sin(phi), np.cos(phi)])

    band_len, band_width = 3.5, 0.35
    corners = np.array([
        theta_star + phys_dir * band_len - phys_normal * band_width,
        theta_star + phys_dir * band_len + phys_normal * band_width,
        theta_star - phys_dir * band_len + phys_normal * band_width,
        theta_star - phys_dir * band_len - phys_normal * band_width,
    ])
    ax.add_patch(plt.Polygon(corners, alpha=0.20, color=C_PHYSICS, linewidth=0))
    for sign in [-1, 1]:
        p1 = theta_star + phys_dir * band_len + sign * phys_normal * band_width
        p2 = theta_star - phys_dir * band_len + sign * phys_normal * band_width
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color=C_PHYSICS, alpha=0.5, lw=0.8)

    label_pos = theta_star + phys_dir * 2.0 + phys_normal * 0.65
    ax.text(label_pos[0], label_pos[1], 'Physics-Critical\nSubspace',
            fontsize=7, color=C_PHYSICS, ha='center', va='bottom',
            fontstyle='italic', fontweight='bold', rotation=25)

    null_label_pos = theta_star + phys_normal * 2.2 - phys_dir * 0.3
    ax.text(null_label_pos[0], null_label_pos[1], 'Null Space\n(Free to Adapt)',
            fontsize=7, color='#27ae60', ha='center', va='bottom',
            fontstyle='italic', fontweight='bold')

    ax.plot(*theta_star, 'ko', markersize=7, zorder=10)
    ax.annotate(r'$\theta^*$', theta_star, textcoords="offset points",
                xytext=(-14, -10), fontsize=11, fontweight='bold', zorder=10)

    # DFT
    dft_end = theta_star + np.array([1.8, 1.6])
    ax.annotate('', xy=dft_end, xytext=theta_star,
                arrowprops=dict(arrowstyle='->', color=C_DFT, lw=2.2))
    ax.text(dft_end[0] + 0.1, dft_end[1] + 0.1, r'$\theta_{\mathrm{DFT}}$',
            fontsize=9, color=C_DFT, fontweight='bold')
    cross_pos = theta_star + np.array([0.9, 0.8])
    ax.plot(cross_pos[0], cross_pos[1], 'x', color=C_DFT, markersize=10, markeredgewidth=2.5, zorder=5)
    ax.text(cross_pos[0] + 0.25, cross_pos[1] - 0.25, 'Catastrophic\nForgetting',
            fontsize=6.5, color=C_DFT, ha='left', va='top', fontstyle='italic')

    # EWC
    ewc_end = theta_star + np.array([1.4, 1.2])
    ax.annotate('', xy=ewc_end, xytext=theta_star,
                arrowprops=dict(arrowstyle='->', color=C_EWC, lw=2.0, connectionstyle='arc3,rad=-0.15'))
    ax.text(ewc_end[0] + 0.1, ewc_end[1] + 0.1, r'$\theta_{\mathrm{EWC}}$',
            fontsize=9, color=C_EWC, fontweight='bold')
    ax.text(theta_star[0] + 0.1, theta_star[1] + 0.7, 'Soft Penalty',
            fontsize=6.5, color=C_EWC, ha='center', fontstyle='italic')

    # PhysGuard
    nsft_end = theta_star + phys_normal * 1.8
    ax.annotate('', xy=nsft_end, xytext=theta_star,
                arrowprops=dict(arrowstyle='->', color=C_NSFT, lw=2.5))
    ax.text(nsft_end[0] - 0.2, nsft_end[1] + 0.15, r'$\theta_{\mathrm{PhysGuard}}$',
            fontsize=9, color=C_NSFT, fontweight='bold')
    ax.text(nsft_end[0] + 0.5, nsft_end[1] - 0.15, 'Physics\nPreserved',
            fontsize=6.5, color=C_NSFT, ha='left', fontstyle='italic', fontweight='bold')

    rangle_size = 0.2
    rp1 = theta_star + phys_dir * rangle_size
    rp2 = theta_star + phys_dir * rangle_size + phys_normal * rangle_size
    rp3 = theta_star + phys_normal * rangle_size
    ax.plot([rp1[0], rp2[0], rp3[0]], [rp1[1], rp2[1], rp3[1]], '-', color=C_NSFT, lw=1.0)

    ax.set_title('(a) Parameter-Space Geometry', fontsize=10, fontweight='bold', pad=8)


# ============================================================================
# Panel (b): Flow Field via cropping existing eval images
# ============================================================================
def crop_panel(img_path, row_type='pred', col_idx=COL_IDX):
    """Crop a specific subplot from a vorticity visualization image."""
    img = Image.open(img_path)
    r_top, r_bot = CROP_ROWS[row_type]
    c_left, c_right = CROP_COLS[col_idx]
    return img.crop((c_left, r_top, c_right, r_bot))


def draw_flow_comparison(ax_array):
    """
    Draw 4-row flow field comparison: GT, Pretrained pred, DFT pred, PhysGuard pred.
    Cropped from existing eval vorticity images with precise pixel coordinates.
    """
    panels = [
        ('Real PIV (Ground Truth)', VORT_IMGS['Pretrained'], 'gt'),
        ('Pretrained (Sim-only)',   VORT_IMGS['Pretrained'], 'pred'),
        ('DFT (Fine-tuned)',        VORT_IMGS['DFT'],        'pred'),
        ('PhysGuard (Ours)',        VORT_IMGS['PhysGuard'],  'pred'),
    ]
    border_colors = ['k', C_PRETRAINED, C_DFT, C_NSFT]

    for idx, (label, img_path, row_type) in enumerate(panels):
        ax = ax_array[idx]
        bc = border_colors[idx]
        if os.path.exists(img_path):
            panel_img = crop_panel(img_path, row_type)
            ax.imshow(panel_img, aspect='auto')
        else:
            ax.text(0.5, 0.5, f'{label}\n(image not found)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='gray')

        ax.set_ylabel(label, fontsize=7.5, fontweight='bold', rotation=90, labelpad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(bc)
            spine.set_linewidth(2.0 if idx > 0 else 1.0)

    ax_array[0].set_title('(b) Vorticity Field (FNO, Cylinder)', fontsize=10, fontweight='bold', pad=8)


# ============================================================================
# Panel (c): Frequency bar chart
# ============================================================================
def draw_freq_barchart(ax):
    architectures = ['FNO', 'CNO', 'DPOT', 'DeepONet']
    methods = ['Pretrained', 'DFT', 'EWC-FT', 'PhysGuard']
    colors = [C_PRETRAINED, C_DFT, C_EWC, C_NSFT]
    hatches = ['', '//', '\\\\', '']

    low_f = {m: [FREQ_DATA[a][m][0] for a in architectures] for m in methods}

    x = np.arange(len(architectures))
    bar_w = 0.18
    offsets = np.arange(len(methods)) * bar_w - (len(methods) - 1) * bar_w / 2

    for i, method in enumerate(methods):
        bars = ax.bar(x + offsets[i], low_f[method], bar_w,
                      label=method, color=colors[i], edgecolor='white',
                      linewidth=0.5, alpha=0.9, hatch=hatches[i], zorder=3)
        if method == 'PhysGuard':
            for bar in bars:
                bar.set_edgecolor(C_NSFT)
                bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(architectures, fontsize=8)
    ax.set_ylabel('Low-Freq fRMSE  (lower is better)', fontsize=9)
    ax.set_title('(c) Low-Frequency Error\n(Physics Preservation)', fontsize=10, fontweight='bold', pad=8)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray',
              fontsize=7, ncol=1, columnspacing=0.5, handlelength=1.2)
    ax.set_ylim(0, max(max(v) for v in low_f.values()) * 1.25)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fno_nsft_val = FREQ_DATA['FNO']['PhysGuard'][0]
    ax.annotate(f'{fno_nsft_val:.4f}',
                xy=(offsets[3], fno_nsft_val),
                xytext=(0.6, fno_nsft_val + 0.005),
                fontsize=7, color=C_NSFT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_NSFT, lw=1.2),
                ha='center')


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("Generating Figure 1: PhysGuard Motivation Figure")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 5.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 3.5, 3.5],
                  wspace=0.08, left=0.02, right=0.98, top=0.92, bottom=0.08)

    ax_concept = fig.add_subplot(gs[0])
    draw_concept_diagram(ax_concept)

    gs_b = gs[1].subgridspec(4, 1, hspace=0.08)
    ax_flow = [fig.add_subplot(gs_b[i]) for i in range(4)]
    draw_flow_comparison(ax_flow)

    ax_freq = fig.add_subplot(gs[2])
    draw_freq_barchart(ax_freq)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUTPUT_DIR, 'figure1_motivation.pdf')
    out_png = os.path.join(OUTPUT_DIR, 'figure1_motivation.png')
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0.05)
    print(f"\nFigure saved to:\n  {out_pdf}\n  {out_png}")
    plt.close(fig)


if __name__ == '__main__':
    main()
