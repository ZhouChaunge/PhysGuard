#!/usr/bin/env python3
"""
RQ1 Unified Summary Figure — Framework B (3 panels)
=====================================================
Framework: Fix architecture (FNO), compare three physical scenarios.
Core claim: FIM principal subspace consistently aligns with low-frequency
            physics modes in laminar/transitional flows, but NOT in
            turbulent reactive flows (combustion), reflecting the
            intrinsic spectral character of each physical regime.

Panels:
  (a) Spectral rank heatmap — FNO / Cylinder
      Energy migrates from low-k to high-k as FIM rank increases.

  (b) f_low vs FIM rank — FNO on Cylinder + Ctrl-Cylinder
      Both show strong monotone decline (ρ ≈ −0.80, ***).

  (c) f_low vs FIM rank — FNO on Turbulent Combustion
      No significant trend (ρ = −0.43, n.s.): flame fronts are
      inherently high-frequency, so FIM directions encode high-freq
      physics — consistent with the hypothesis.

Run:
  python \\
      ./scripts/rq1_unified_summary_figure.py
"""

import numpy as np
import os
from scipy.stats import spearmanr

OUTPUT_DIR = "./figures"

# ── Load data ─────────────────────────────────────────────────────

# Panel (a): heatmap — FNO / Cylinder, 3D FFT
d_heat      = np.load(f"{OUTPUT_DIR}/rq1_fim_intuitive_data.npz", allow_pickle=True)
spectra_raw = d_heat["spectra_matrix"]   # [20, half]
half_a      = int(d_heat["half"])

# Panels (b) & (c): f_low sorted by FIM rank j (2D spatial FFT, all three datasets)
d_fno      = np.load(f"{OUTPUT_DIR}/rq1_random_baseline_data.npz")
f_low_cyl  = d_fno["fl_top"]        # shape=(50,), ranks 1…50
f_low_rnd  = d_fno["fl_random"]

d_cc       = np.load(f"{OUTPUT_DIR}/rq1_fno_ctrl_cylinder_2dfft_data.npz",
                     allow_pickle=True)
f_low_cc   = d_cc["f_low"][:20]     # shape=(20,), ranks 1…20

d_comb     = np.load(f"{OUTPUT_DIR}/rq1_fno_combustion_2dfft_data.npz",
                     allow_pickle=True)
f_low_comb = d_comb["f_low"][:20]   # shape=(20,), ranks 1…20

# Spearman ρ — computed from data
K_cyl  = len(f_low_cyl)   # 50
K_cc   = len(f_low_cc)    # 20
K_comb = len(f_low_comb)  # 20
rho_cyl,  pval_cyl  = spearmanr(np.arange(1, K_cyl  + 1), f_low_cyl)
rho_cc,   pval_cc   = spearmanr(np.arange(1, K_cc   + 1), f_low_cc)
rho_comb, pval_comb = spearmanr(np.arange(1, K_comb + 1), f_low_comb)
rnd_mean = f_low_rnd.mean()

print(f"Cylinder:      ρ={rho_cyl:+.3f}, p={pval_cyl:.2e}  (N={K_cyl})")
print(f"Ctrl-Cylinder: ρ={rho_cc:+.3f}, p={pval_cc:.2e}  (N={K_cc})")
print(f"Combustion:    ρ={rho_comb:+.3f}, p={pval_comb:.2e}  (N={K_comb})")
print(f"Random baseline: mean={rnd_mean:.4f}")


# ── Helper ────────────────────────────────────────────────────────

def sig_stars(p):
    if   p < 0.001: return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "n.s."


# ── Figure ────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import uniform_filter1d

FIG_W, FIG_H = 14.0, 4.6

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs  = gridspec.GridSpec(
    1, 3,
    width_ratios=[1.15, 1.45, 1.0],
    wspace=0.38,
    left=0.055, right=0.97,
    top=0.86,   bottom=0.15,
)

DS_COLORS = {
    "Cylinder":      "#4C72B0",
    "Ctrl-Cylinder": "#E07B54",
    "Combustion":    "#55A868",
}

# ═══════════════════════════════════════════════════════════════════
# Panel (a): Spectral rank heatmap — FNO / Cylinder
# ═══════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])

K_heat = spectra_raw.shape[0]   # 20
k_show = half_a                  # 10
iLow_a = int(round(half_a / 3))

spec_norm = spectra_raw.copy().astype(float)
for j in range(K_heat):
    mx = spec_norm[j].max()
    if mx > 0:
        spec_norm[j] /= mx

spec_sm = uniform_filter1d(spec_norm, size=2, axis=1)
img = spec_sm[:, :k_show].T    # [k_show, K_heat]

im = ax_a.imshow(
    img, aspect="auto", origin="lower",
    extent=[0.5, K_heat + 0.5, -0.5, k_show - 0.5],
    cmap="RdYlBu_r", vmin=0, vmax=1,
    interpolation="bilinear",
)
ax_a.axhline(iLow_a - 0.5, color="white", lw=1.8, ls="--", alpha=0.88)

ax_a.set_xlabel("FIM rank $j$  (1 = most important)", fontsize=10.5)
ax_a.set_ylabel("Radial wavenumber $k$", fontsize=10.5)
ax_a.set_title("(a)  Spectral energy of $\\Delta y$ per FIM direction\n"
               r"(FNO · Cylinder Flow)",
               fontsize=10.5, pad=5)
ax_a.set_xticks([1, 5, 10, 15, 20])
ax_a.yaxis.set_major_locator(MaxNLocator(integer=True))

cb = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
cb.set_label("Norm. energy", fontsize=9)

ax_a.text(0.04, 0.09, "Low-freq\n(physics)", transform=ax_a.transAxes,
          fontsize=8, color="white", va="bottom",
          bbox=dict(fc="#1f77b4", ec="none", alpha=0.75, pad=2))
ax_a.text(0.04, 0.62, "High-freq\n(noise)", transform=ax_a.transAxes,
          fontsize=8, color="white", va="bottom",
          bbox=dict(fc="#d62728", ec="none", alpha=0.70, pad=2))

# ═══════════════════════════════════════════════════════════════════
# Panel (b): f_low vs rank — Cylinder + Ctrl-Cylinder (positive evidence)
# ═══════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1])

ranks_cyl = np.arange(1, K_cyl + 1)
ranks_cc  = np.arange(1, K_cc  + 1)

z_cyl  = np.polyfit(ranks_cyl, f_low_cyl, 1)
z_cc   = np.polyfit(ranks_cc,  f_low_cc,  1)
tr_cyl = np.polyval(z_cyl, ranks_cyl)
tr_cc  = np.polyval(z_cc,  ranks_cc)

ax_b.fill_between(ranks_cyl, tr_cyl, tr_cyl[0] + 5e-4,
                  alpha=0.12, color=DS_COLORS["Cylinder"], zorder=1)
ax_b.fill_between(ranks_cc,  tr_cc,  tr_cc[0]  + 5e-4,
                  alpha=0.12, color=DS_COLORS["Ctrl-Cylinder"], zorder=1)

ax_b.scatter(ranks_cyl, f_low_cyl, color=DS_COLORS["Cylinder"],
             s=26, alpha=0.65, zorder=3,
             label="Cylinder flow  ($N=50$ ranks)")
ax_b.scatter(ranks_cc, f_low_cc, color=DS_COLORS["Ctrl-Cylinder"], marker="s",
             s=26, alpha=0.65, zorder=3,
             label="Controlled cylinder  ($N=20$ ranks)")
ax_b.plot(ranks_cyl, tr_cyl, color=DS_COLORS["Cylinder"],  lw=2.2, zorder=4)
ax_b.plot(ranks_cc,  tr_cc,  color=DS_COLORS["Ctrl-Cylinder"], lw=2.2, ls="--", zorder=4)

stars_cyl = sig_stars(pval_cyl)
stars_cc  = sig_stars(pval_cc)
ax_b.text(0.97, 0.96,
          r"$\rho=%+.2f^{\,%s}$" % (rho_cyl, stars_cyl),
          transform=ax_b.transAxes, fontsize=10,
          color=DS_COLORS["Cylinder"], ha="right", va="top", fontweight="bold")
ax_b.text(0.97, 0.84,
          r"$\rho=%+.2f^{\,%s}$" % (rho_cc, stars_cc),
          transform=ax_b.transAxes, fontsize=10,
          color=DS_COLORS["Ctrl-Cylinder"], ha="right", va="top", fontweight="bold")
ax_b.text(0.02, 0.04,
          r"Random: $f_{\mathrm{low}}\approx%.2f$  ↓" % rnd_mean,
          transform=ax_b.transAxes, fontsize=8.5,
          color="gray", ha="left", va="bottom", style="italic")

ax_b.set_xlabel("FIM rank $j$  (1 = most important)", fontsize=10.5)
ax_b.set_ylabel(r"$f_{\mathrm{low}}$  (low-freq energy fraction)", fontsize=10.5)
ax_b.set_title(
    r"(b)  $f_{\mathrm{low}}$ decreases with FIM rank"
    "\n(Laminar / transitional flows  →  low-freq physics)",
    fontsize=10.5, pad=5)
ax_b.set_xlim(0, K_cyl + 1)
y_b = np.concatenate([f_low_cyl, f_low_cc])
y_br = y_b.max() - y_b.min()
ax_b.set_ylim(y_b.min() - y_br * 0.25, y_b.max() + y_br * 0.45)
ax_b.legend(loc="lower left", fontsize=8.5, framealpha=0.92,
            edgecolor="lightgray", handlelength=1.6)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.yaxis.grid(True, linestyle="--", alpha=0.35)
ax_b.set_axisbelow(True)

# ═══════════════════════════════════════════════════════════════════
# Panel (c): Turbulent combustion — contrast / negative case
# ═══════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[2])

ranks_comb = np.arange(1, K_comb + 1)
z_comb  = np.polyfit(ranks_comb, f_low_comb, 1)
tr_comb = np.polyval(z_comb, ranks_comb)

ax_c.scatter(ranks_comb, f_low_comb, color=DS_COLORS["Combustion"], marker="D",
             s=28, alpha=0.70, zorder=3, label="Turbulent combustion")
ax_c.plot(ranks_comb, tr_comb, color=DS_COLORS["Combustion"],
          lw=1.8, ls=":", zorder=4, alpha=0.75)

# Shade the "expected" zone for reference (region occupied by Cyl/CtrlCyl)
cyl_lo = min(f_low_cyl.min(), f_low_cc.min())
cyl_hi = max(f_low_cyl.max(), f_low_cc.max())
ax_c.axhspan(cyl_lo, cyl_hi, alpha=0.08, color="#888888",
             label=r"Cylinder / Ctrl-Cyl range")

stars_comb = sig_stars(pval_comb)
ax_c.text(0.97, 0.96,
          r"$\rho=%+.2f$, %s" % (rho_comb, stars_comb),
          transform=ax_c.transAxes, fontsize=10,
          color=DS_COLORS["Combustion"], ha="right", va="top", fontweight="bold")

# Annotation explaining why
ax_c.text(0.50, 0.06,
          "Flame fronts are\ninherently high-freq",
          transform=ax_c.transAxes, fontsize=8, ha="center", va="bottom",
          color=DS_COLORS["Combustion"],
          bbox=dict(fc="white", ec=DS_COLORS["Combustion"],
                    alpha=0.85, pad=2.5, lw=0.8, boxstyle="round,pad=0.3"))

ax_c.set_xlabel("FIM rank $j$  (1 = most important)", fontsize=10.5)
ax_c.set_ylabel(r"$f_{\mathrm{low}}$", fontsize=10.5)
ax_c.set_title(
    r"(c)  No trend in turbulent combustion"
    "\n(Reactive flow  →  high-freq physics)",
    fontsize=10.5, pad=5)
ax_c.set_xlim(0, K_comb + 1)
y_c = f_low_comb
y_cr = y_c.max() - y_c.min()
ax_c.set_ylim(y_c.min() - y_cr * 0.15, y_c.max() + y_cr * 0.60)
ax_c.legend(loc="upper left", fontsize=8, framealpha=0.88,
            edgecolor="lightgray", handlelength=1.4)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.yaxis.grid(True, linestyle="--", alpha=0.35)
ax_c.set_axisbelow(True)

# ── Shared super-title ────────────────────────────────────────────
fig.suptitle(
    r"FIM principal subspace encodes low-frequency physics modes"
    r" — pattern holds across flows, breaks for reactive turbulence"
    "\n"
    r"(metric: $f_{\mathrm{low}}$ = fraction of $\|\Delta y\|^2$ in low spatial wavenumbers  |  FNO · 2D spatial FFT)",
    fontsize=10.5, y=1.02,
)

# ── Save ──────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
for ext in ["pdf", "png"]:
    path = f"{OUTPUT_DIR}/rq1_unified_summary.{ext}"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {path}")

plt.close(fig)

print()
print(f"  Cylinder:      ρ={rho_cyl:+.3f}  p={pval_cyl:.2e}  {sig_stars(pval_cyl)}")
print(f"  Ctrl-Cylinder: ρ={rho_cc:+.3f}  p={pval_cc:.2e}  {sig_stars(pval_cc)}")
print(f"  Combustion:    ρ={rho_comb:+.3f}  p={pval_comb:.2e}  {sig_stars(pval_comb)}")


