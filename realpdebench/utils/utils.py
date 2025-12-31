import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm
from torch.utils.tensorboard import SummaryWriter


# config utils
def add_args_from_config(args, parser=None):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Determine which args were explicitly passed on the command line
    if parser is not None:
        defaults = vars(parser.parse_args([]))
        cli_explicit = {k for k, v in vars(args).items() if v != defaults.get(k)}
    else:
        cli_explicit = set(vars(args).keys())
    
    for key, value in config.items():
        if key in cli_explicit:
            # CLI-explicit args take precedence, don't override
            continue
        setattr(args, key, value)
    return args


# training utils
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(exp_path, is_use_tb=False, is_train=True):
    if is_train:
        log_filename = os.path.join(exp_path, f"training.log")
    else:
        log_filename = os.path.join(exp_path, f"eval.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized at {log_filename}")

    if is_use_tb:
        writer = SummaryWriter(log_dir=exp_path)
        logging.info(f"Tensorboard writer initialized at {writer.log_dir}")
    else:
        writer = None
        
    return writer

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# test utils
# Channel metadata for fluid flow: channel index -> (name, colormap, symmetric)
_FLUID_CHANNEL_META = {
    0: ("u (x-velocity)",  "RdBu_r",  True),
    1: ("v (y-velocity)",  "RdBu_r",  True),
    2: ("p (pressure)",    "viridis",  False),
}


def _make_contourf(ax, field, cmap, symmetric, n_levels=64, title=""):
    """Draw a contourf flow-field panel on ax and return the QuadContourSet."""
    h, w = field.shape
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    X, Y = np.meshgrid(x, y)

    vmin, vmax = field.min(), field.max()
    if symmetric:
        # Use symmetric limits so zero is always white/neutral
        abs_max = max(abs(vmin), abs(vmax), 1e-8)
        vmin, vmax = -abs_max, abs_max

    levels = np.linspace(vmin, vmax, n_levels)
    cf = ax.contourf(X, Y, field, levels=levels, cmap=cmap, extend="both")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=3)
    return cf


def _add_streamlines(ax, u_field, v_field, density=1.0, color="k", linewidth=0.5):
    """Overlay velocity streamlines on an existing axis."""
    h, w = u_field.shape
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    try:
        ax.streamplot(x, y, u_field, v_field,
                      density=density, color=color,
                      linewidth=linewidth, arrowsize=0.6)
    except Exception:
        pass  # streamplot may fail on degenerate fields; silently skip


def plot_result(pred, target, exp_path, N_plot, unmeasured_c, draw_streamlines=True):
    """
    Visualise flow-field predictions as contourf cloud maps (云图).

    Layout per sample per channel:
      Row 0: Ground Truth  (4 time snapshots)
      Row 1: Prediction    (4 time snapshots)
      Row 2: |Error|       (4 time snapshots)

    For velocity channels (u, v), streamlines are optionally overlaid on
    the Ground Truth and Prediction panels.

    Extra figure: if both u (ch 0) and v (ch 1) are present, a velocity-
    magnitude + streamline overview is saved as `streamline_{idx}.png`.

    Args:
        pred / target : torch.Tensor or np.ndarray, shape [b, t, h, w, c]
        exp_path      : output directory (a sub-dir "figs/" is created inside)
        N_plot        : number of samples to visualise (≤ b)
        unmeasured_c  : number of trailing channels to skip (e.g. masked pressure)
        draw_streamlines : whether to overlay streamlines on u/v panels
    """
    save_dir = os.path.join(exp_path, "figs")
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    b, t_, h, w, c_total = pred.shape
    N_plot = min(N_plot, b)
    c = c_total - unmeasured_c  # number of channels to actually plot

    # Choose 4 representative time indices spread across the horizon
    t_indices = [t_ // 4 * k + (t_ - 1) % 4 for k in range(4)]

    for idx in tqdm.tqdm(range(N_plot), desc="Plotting flow-field results"):
        # ------------------------------------------------------------------ #
        # Per-channel contourf 云图                                           #
        # ------------------------------------------------------------------ #
        for ch in range(c):
            meta = _FLUID_CHANNEL_META.get(ch, (f"channel {ch}", "viridis", False))
            ch_name, cmap, symmetric = meta

            fig, axes = plt.subplots(3, 4, figsize=(18, 9))
            fig.suptitle(f"Sample {idx} — {ch_name}", fontsize=12, y=1.01)

            row_labels = ["Ground Truth", "Prediction", "|Error|"]

            for col, t in enumerate(t_indices):
                gt_field   = target[idx, t, :, :, ch]
                pred_field = pred[idx, t, :, :, ch]
                err_field  = np.abs(pred_field - gt_field)

                # Ground Truth
                cf_gt = _make_contourf(axes[0, col], gt_field, cmap, symmetric,
                                       title=f"GT  t={t}")
                fig.colorbar(cf_gt, ax=axes[0, col], fraction=0.03, pad=0.02,
                             format="%.2g")
                if draw_streamlines and ch in (0, 1):
                    u_gt = target[idx, t, :, :, 0]
                    v_gt = target[idx, t, :, :, 1]
                    _add_streamlines(axes[0, col], u_gt, v_gt, density=0.8)

                # Prediction
                cf_pr = _make_contourf(axes[1, col], pred_field, cmap, symmetric,
                                       title=f"Pred t={t}")
                fig.colorbar(cf_pr, ax=axes[1, col], fraction=0.03, pad=0.02,
                             format="%.2g")
                if draw_streamlines and ch in (0, 1):
                    u_pr = pred[idx, t, :, :, 0]
                    v_pr = pred[idx, t, :, :, 1]
                    _add_streamlines(axes[1, col], u_pr, v_pr, density=0.8)

                # Error (always use sequential colormap)
                cf_er = _make_contourf(axes[2, col], err_field, "hot_r", False,
                                       title=f"Error t={t}")
                fig.colorbar(cf_er, ax=axes[2, col], fraction=0.03, pad=0.02,
                             format="%.2g")

            # Add row labels on the left
            for row, label in enumerate(row_labels):
                axes[row, 0].set_ylabel(label, fontsize=10, labelpad=4)

            plt.tight_layout()
            safe_name = ch_name.split(" ")[0].replace("/", "_")
            fname = f"sample{idx:03d}_{safe_name}_contourf.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=120, bbox_inches="tight")
            plt.close(fig)

        # ------------------------------------------------------------------ #
        # Velocity magnitude + streamline overview (when u and v both exist)  #
        # ------------------------------------------------------------------ #
        if c >= 2:
            n_cols = min(4, len(t_indices))
            fig2, axes2 = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
            if n_cols == 1:
                axes2 = axes2[:, None]
            fig2.suptitle(f"Sample {idx} — Velocity Magnitude & Streamlines",
                          fontsize=12, y=1.01)

            for col, t in enumerate(t_indices[:n_cols]):
                u_gt   = target[idx, t, :, :, 0]
                v_gt   = target[idx, t, :, :, 1]
                u_pr   = pred[idx, t, :, :, 0]
                v_pr   = pred[idx, t, :, :, 1]
                mag_gt = np.sqrt(u_gt**2 + v_gt**2)
                mag_pr = np.sqrt(u_pr**2 + v_pr**2)

                vmax = max(mag_gt.max(), mag_pr.max(), 1e-8)

                for row, (mag, uf, vf, lbl) in enumerate([
                        (mag_gt, u_gt, v_gt, f"GT  t={t}"),
                        (mag_pr, u_pr, v_pr, f"Pred t={t}")]):
                    ax = axes2[row, col]
                    x = np.linspace(0, mag.shape[1] - 1, mag.shape[1])
                    y = np.linspace(0, mag.shape[0] - 1, mag.shape[0])
                    X, Y = np.meshgrid(x, y)
                    cf = ax.contourf(X, Y, mag, levels=64,
                                     cmap="jet", vmin=0, vmax=vmax)
                    fig2.colorbar(cf, ax=ax, fraction=0.03, pad=0.02,
                                  label="|U|", format="%.2g")
                    _add_streamlines(ax, uf, vf, density=1.2,
                                     color="white", linewidth=0.7)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(lbl, fontsize=9)

            axes2[0, 0].set_ylabel("Ground Truth", fontsize=10)
            axes2[1, 0].set_ylabel("Prediction",   fontsize=10)
            plt.tight_layout()
            fig2.savefig(os.path.join(save_dir, f"sample{idx:03d}_streamlines.png"),
                         dpi=120, bbox_inches="tight")
            plt.close(fig2)

        # ------------------------------------------------------------------ #
        # Vorticity contourf (computed from u, v if available)                #
        # ------------------------------------------------------------------ #
        if c >= 2:
            fig3, axes3 = plt.subplots(3, 4, figsize=(18, 9))
            fig3.suptitle(f"Sample {idx} — Vorticity (∂v/∂x − ∂u/∂y)",
                          fontsize=12, y=1.01)

            for col, t in enumerate(t_indices):
                u_gt = target[idx, t, :, :, 0]
                v_gt = target[idx, t, :, :, 1]
                u_pr = pred[idx, t, :, :, 0]
                v_pr = pred[idx, t, :, :, 1]

                # Finite-difference vorticity: ∂v/∂x − ∂u/∂y
                vort_gt = np.gradient(v_gt, axis=1) - np.gradient(u_gt, axis=0)
                vort_pr = np.gradient(v_pr, axis=1) - np.gradient(u_pr, axis=0)
                vort_er = np.abs(vort_pr - vort_gt)

                cf0 = _make_contourf(axes3[0, col], vort_gt, "RdBu_r", True,
                                     title=f"GT  t={t}")
                fig3.colorbar(cf0, ax=axes3[0, col], fraction=0.03, pad=0.02,
                              format="%.2g")

                cf1 = _make_contourf(axes3[1, col], vort_pr, "RdBu_r", True,
                                     title=f"Pred t={t}")
                fig3.colorbar(cf1, ax=axes3[1, col], fraction=0.03, pad=0.02,
                              format="%.2g")

                cf2 = _make_contourf(axes3[2, col], vort_er, "hot_r", False,
                                     title=f"Error t={t}")
                fig3.colorbar(cf2, ax=axes3[2, col], fraction=0.03, pad=0.02,
                              format="%.2g")

            for row, label in enumerate(["GT Vorticity", "Pred Vorticity",
                                          "|Vort Error|"]):
                axes3[row, 0].set_ylabel(label, fontsize=10, labelpad=4)

            plt.tight_layout()
            fig3.savefig(os.path.join(save_dir, f"sample{idx:03d}_vorticity.png"),
                         dpi=120, bbox_inches="tight")
            plt.close(fig3)

    logging.info(f"Flow-field visualizations saved at {save_dir}")


def minimal_plot_result(pred, target, exp_path, N_plot, unmeasured_c):
    """
    Minimal publication-quality flow-field visualization.

    For each sample, for each time step, saves a *separate* figure with:
      - Left: Ground Truth  (velocity magnitude + streamlines)
      - Right: Prediction   (velocity magnitude + streamlines)

    Each figure is a standalone PDF/PNG, easy to crop/compose in LaTeX.
    Uses jet colormap + white streamlines, shared colorbar.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 9,
    })

    save_dir = os.path.join(exp_path, "figs_minimal")
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    b, t_, h, w, c_total = pred.shape
    N_plot = min(N_plot, b)
    c = c_total - unmeasured_c

    # 4 representative time indices
    t_indices = [t_ // 4 * k + (t_ - 1) % 4 for k in range(4)]

    if c < 2:
        logging.warning("minimal_plot_result requires at least 2 velocity channels (u, v).")
        return

    for idx in tqdm.tqdm(range(N_plot), desc="Minimal flow-field plots"):
        for t in t_indices:
            u_gt = target[idx, t, :, :, 0]
            v_gt = target[idx, t, :, :, 1]
            u_pr = pred[idx, t, :, :, 0]
            v_pr = pred[idx, t, :, :, 1]
            mag_gt = np.sqrt(u_gt**2 + v_gt**2)
            mag_pr = np.sqrt(u_pr**2 + v_pr**2)

            vmax = max(mag_gt.max(), mag_pr.max(), 1e-8)

            x = np.linspace(0, w - 1, w)
            y = np.linspace(0, h - 1, h)
            X, Y = np.meshgrid(x, y)

            fig, axes = plt.subplots(1, 2, figsize=(8, 2.4))

            for col, (mag, uf, vf, lbl) in enumerate([
                    (mag_gt, u_gt, v_gt, "Ground Truth"),
                    (mag_pr, u_pr, v_pr, "Prediction")]):
                ax = axes[col]
                cf = ax.contourf(X, Y, mag, levels=64,
                                 cmap="jet", vmin=0, vmax=vmax)
                _add_streamlines(ax, uf, vf, density=1.0,
                                 color="white", linewidth=0.5)
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(lbl, fontsize=9, fontweight='bold', pad=3)

            # Single shared colorbar on the right
            fig.subplots_adjust(right=0.88, wspace=0.06)
            cax = fig.add_axes([0.89, 0.15, 0.015, 0.7])
            cb = fig.colorbar(plt.cm.ScalarMappable(
                norm=Normalize(0, vmax), cmap="jet"), cax=cax)
            cb.ax.tick_params(labelsize=7)
            cb.set_label(r'$|\mathbf{u}|$', fontsize=9)

            # Save as both PDF and PNG
            base_name = f"sample{idx:03d}_t{t:03d}"
            fig.savefig(os.path.join(save_dir, f"{base_name}.pdf"),
                        dpi=300, bbox_inches="tight", pad_inches=0.02)
            fig.savefig(os.path.join(save_dir, f"{base_name}.png"),
                        dpi=300, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

    logging.info(f"Minimal flow-field visualizations saved at {save_dir}")