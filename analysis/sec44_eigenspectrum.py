"""
§4.4 Fisher Subspace Eigenvalue Spectrum Analysis
==================================================
For each architecture (FNO, CNO, DeepONet, Transolver):
  1. Load pretrained model + numerical data
  2. Collect per-sample gradients → build Gram matrix K per layer
  3. Extract FULL eigenvalue spectrum {λ_j}
  4. Save to analysis/figures/eigenspectrum_data.pt

Then generate figures/005-eigenspectrum.png with 3 subplots:
  (a) Eigenvalue decay (log scale, largest layer per arch)
  (b) Cumulative variance ratio (τ=0.9 line)
  (c) Adaptive k per layer (grouped bar chart)

Usage:
    # Phase A: extract eigenvalues (requires GPU + data)
    python analysis/sec44_eigenspectrum.py --extract --gpu 1

    # Phase B: plot only (from cached data)
    python analysis/sec44_eigenspectrum.py --plot

    # Both:
    python analysis/sec44_eigenspectrum.py --extract --plot --gpu 1
"""

import os
import sys
import gc
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "RealPDEBench"))

CACHE_DIR = PROJECT_ROOT / "analysis" / "figures"
FIG_DIR = PROJECT_ROOT / "figures"

# ── Per-dataset architecture configs ─────────────────────────────────────────
DATASET_CONFIGS = {
    "cylinder": {
        "title": "Cylinder",
        "fig_id": "005",
        "archs": {
            "FNO": {"config": "realpdebench/configs/cylinder/fno_nullspace.yaml", "max_samples": 200},
            "CNO": {"config": "realpdebench/configs/cylinder/cno_nullspace.yaml", "max_samples": 200},
            "DeepONet": {"config": "realpdebench/configs/cylinder/deeponet_nullspace.yaml", "max_samples": 200},
            "Transolver": {"config": "realpdebench/configs/cylinder/transolver_nullspace.yaml", "max_samples": 200},
        },
    },
    "controlled_cylinder": {
        "title": "Controlled Cylinder",
        "fig_id": "005b",
        "archs": {
            "FNO": {"config": "realpdebench/configs/controlled_cylinder/fno_nullspace.yaml", "max_samples": 200},
            "CNO": {"config": "realpdebench/configs/controlled_cylinder/cno_nullspace.yaml", "max_samples": 200},
            "DeepONet": {"config": "realpdebench/configs/controlled_cylinder/deeponet_nullspace.yaml", "max_samples": 200},
            "Transolver": {"config": "realpdebench/configs/controlled_cylinder/transolver_nullspace.yaml", "max_samples": 200},
        },
    },
    "combustion": {
        "title": "Turbulent Combustion",
        "fig_id": "005c",
        "archs": {
            "FNO": {"config": "realpdebench/configs/combustion/fno_nullspace.yaml", "max_samples": 200},
            "CNO": {"config": "realpdebench/configs/combustion/cno_nullspace.yaml", "max_samples": 200},
            "DeepONet": {"config": "realpdebench/configs/combustion/deeponet_nullspace.yaml", "max_samples": 200},
            "Transolver": {"config": "realpdebench/configs/combustion/transolver_nullspace.yaml", "max_samples": 200},
        },
    },
}

VARIANCE_THRESHOLD = 0.9


def extract_eigenvalues_for_arch(arch_name, config_path, max_samples, device):
    """
    Load model + numerical data, collect per-sample gradients,
    build Gram matrix per layer, return dict of eigenvalues.
    """
    from realpdebench.train_nullspace import build_datasets, parser as ns_parser
    from realpdebench.model.load_model import load_model
    from realpdebench.data.data_normalizer import (
        IdentityNormalizer, GaussianNormalizer, RangeNormalizer,
    )
    from realpdebench.utils.utils import add_args_from_config

    # Parse config
    args = ns_parser.parse_args(["--config", config_path])
    args = add_args_from_config(args, ns_parser)

    # Build datasets
    _, _, normalizer_dataset, fisher_dataset, _ = build_datasets(args)

    fisher_dataloader = torch.utils.data.DataLoader(
        fisher_dataset, batch_size=4,
        shuffle=True, pin_memory=True, num_workers=4,
    )

    # Data normalizer
    if args.normalizer == "none":
        data_normalizer = IdentityNormalizer(device=device)
    elif args.normalizer == "gaussian":
        data_normalizer = GaussianNormalizer(normalizer_dataset, device=device)
    elif args.normalizer == "range":
        data_normalizer = RangeNormalizer(normalizer_dataset, device=device)
    else:
        raise ValueError(f"Unknown normalizer: {args.normalizer}")

    # Load model + pretrained checkpoint
    model = load_model(fisher_dataset, device=device, **vars(args))
    assert args.checkpoint_path is not None
    model.load_checkpoint(args.checkpoint_path, device)
    model.eval()
    logging.info(f"[{arch_name}] Model loaded from {args.checkpoint_path}")

    # Collect parameter names
    param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_names.append(name)

    # ── Phase 1a: Collect per-sample gradients ──────────────────────
    grad_accum = {name: [] for name in param_names}
    n_samples = 0

    for batch_data in tqdm(fisher_dataloader, desc=f"[{arch_name}] Collecting gradients"):
        if n_samples >= max_samples:
            break
        input_data, target_data = batch_data
        input_data, target_data = data_normalizer.preprocess(input_data, target_data)
        b = input_data.size(0)

        for i in range(b):
            if n_samples >= max_samples:
                break
            model.zero_grad()
            inp_i = input_data[i : i + 1]
            tgt_i = target_data[i : i + 1]
            loss = model.train_loss(inp_i, tgt_i).mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    g = param.grad.detach().cpu().reshape(-1)
                    if g.is_complex():
                        g = torch.cat([g.real, g.imag], dim=0)
                    grad_accum[name].append(g.half())

            n_samples += 1

    logging.info(f"[{arch_name}] Collected gradients from {n_samples} samples")

    # ── Phase 1b: Gram trick → eigenvalues per layer ────────────────
    layer_eigenvalues = {}  # name -> 1D tensor of eigenvalues (descending)

    sorted_names = sorted(
        param_names,
        key=lambda n: grad_accum[n][0].numel() if grad_accum.get(n) else 0,
        reverse=True,
    )

    for name in tqdm(sorted_names, desc=f"[{arch_name}] Computing eigenvalues"):
        grads = grad_accum.pop(name, None)
        if not grads:
            continue

        G = torch.stack(grads, dim=0).float()  # [N, d]
        del grads
        gc.collect()

        param_dim = G.shape[1]
        if param_dim <= 1:
            del G
            continue

        # Gram matrix K = G @ G^T  [N, N]
        K = G @ G.T
        del G
        gc.collect()

        # Eigendecompose (ascending order)
        eigenvalues, _ = torch.linalg.eigh(K)
        del K
        gc.collect()

        # Flip to descending, clamp negatives
        evals_desc = eigenvalues.flip(0).clamp(min=0.0)
        layer_eigenvalues[name] = evals_desc.cpu()
        logging.info(
            f"  {name}: dim={param_dim}, n_evals={len(evals_desc)}, "
            f"top={evals_desc[0]:.4e}, ratio_top10={evals_desc[:10].sum()/evals_desc.sum():.4f}"
        )

    # Clean up
    del model, data_normalizer, fisher_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    return layer_eigenvalues


def extract_all(gpu_id, dataset_name):
    """Extract eigenvalues for all 4 architectures on a given dataset."""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    os.chdir(PROJECT_ROOT / "RealPDEBench")

    ds_cfg = DATASET_CONFIGS[dataset_name]
    arch_configs = ds_cfg["archs"]

    all_data = {}
    for arch_name, cfg in arch_configs.items():
        logging.info(f"\n{'='*60}\n[{dataset_name}] Extracting eigenvalues for {arch_name}\n{'='*60}")
        evals = extract_eigenvalues_for_arch(
            arch_name, cfg["config"], cfg["max_samples"], device
        )
        all_data[arch_name] = evals

    # Save
    cache_path = CACHE_DIR / f"eigenspectrum_data_{dataset_name}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_data, cache_path)
    logging.info(f"\nEigenvalue data saved to {cache_path}")
    return all_data


def _filter_weight_layers(layer_dict):
    """Keep only main weight parameters (skip bias, batch_norm, temperature, placeholder)."""
    skip_patterns = ('.bias', 'batch_norm', 'bn', '.temperature', 'placeholder', 'ln_')
    return {
        name: evals for name, evals in layer_dict.items()
        if not any(p in name for p in skip_patterns)
    }


def _compute_adaptive_k(evals, threshold=VARIANCE_THRESHOLD):
    """Return adaptive k for a single eigenvalue vector."""
    total = evals.sum()
    if total > 0:
        cum_ratio = evals.cumsum(0) / total
        above = (cum_ratio >= threshold).nonzero(as_tuple=False)
        return int(above[0].item()) + 1 if len(above) > 0 else len(evals)
    return len(evals)


def plot_figure(all_data, dataset_name="cylinder"):
    """Generate the 3-panel eigenspectrum figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds_cfg = DATASET_CONFIGS[dataset_name]
    fig_id = ds_cfg["fig_id"]
    ds_title = ds_cfg["title"]

    ARCH_ORDER = ["FNO", "CNO", "DeepONet", "Transolver"]
    colors = [plt.cm.tab10(i) for i in range(4)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.rcParams.update({"font.size": 10})

    # ── Pre-compute: for each arch, find the layer with highest top eigenvalue ──
    # (proxy for "most important" / largest Fisher information)
    largest_layer = {}   # arch -> (name, evals)
    weight_layers_k = {} # arch -> list of (name, k)  — weight layers only

    for arch in ARCH_ORDER:
        layers = all_data[arch]
        # Find layer with largest leading eigenvalue
        best_name, best_evals = max(layers.items(), key=lambda x: x[1][0].item() if len(x[1]) > 0 else 0)
        largest_layer[arch] = (best_name, best_evals)

        # Compute adaptive k for weight layers only (sorted by layer name)
        wl = _filter_weight_layers(layers)
        weight_layers_k[arch] = [
            (name, _compute_adaptive_k(evals))
            for name, evals in sorted(wl.items())
        ]

    # ── (a) Eigenvalue decay curves (log scale) ────────────────────
    ax = axes[0]
    for i, arch in enumerate(ARCH_ORDER):
        lname, evals = largest_layer[arch]
        evals_np = evals.numpy()
        nonzero = evals_np > 0
        idx = np.arange(1, len(evals_np) + 1)
        ax.semilogy(idx[nonzero], evals_np[nonzero], color=colors[i],
                     linewidth=1.5, label=arch)
    ax.set_xlabel("Eigenvalue index $j$")
    ax.set_ylabel("$\\lambda_j$")
    ax.set_title("Eigenvalue decay")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

    # ── (b) Cumulative variance ratio ──────────────────────────────
    ax = axes[1]
    k_star_list = []
    for i, arch in enumerate(ARCH_ORDER):
        lname, evals = largest_layer[arch]
        evals_np = evals.numpy()
        total = evals_np.sum()
        cum_ratio = np.cumsum(evals_np) / total if total > 0 else np.ones_like(evals_np)
        k_vals = np.arange(1, len(cum_ratio) + 1)
        ax.plot(k_vals, cum_ratio, color=colors[i], linewidth=1.5, label=arch)

        above = np.where(cum_ratio >= VARIANCE_THRESHOLD)[0]
        if len(above) > 0:
            k_star = above[0] + 1
            k_star_list.append((arch, i, k_star))

    # Smart annotation positioning (avoid overlap)
    y_offsets = [0.82, 0.74, 0.66, 0.58]
    for idx, (arch, ci, k_star) in enumerate(sorted(k_star_list, key=lambda x: x[2])):
        ax.annotate(
            f"$k$={k_star}",
            xy=(k_star, VARIANCE_THRESHOLD),
            xytext=(k_star + 8, y_offsets[idx]),
            fontsize=8, color=colors[ci], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=colors[ci], lw=0.8),
        )

    ax.axhline(y=VARIANCE_THRESHOLD, color="red", linestyle="--", linewidth=1.0,
               label=f"$\\tau$={VARIANCE_THRESHOLD}")
    ax.set_xlabel("$k$")
    ax.set_ylabel("Cumulative ratio $\\rho(k)$")
    ax.set_title("Cumulative variance ratio")
    ax.legend(fontsize=9, framealpha=0.8, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

    # ── (c) Adaptive k per layer (grouped bar chart, weight layers only) ──
    ax = axes[2]
    max_wl = max(len(weight_layers_k[a]) for a in ARCH_ORDER)
    n_archs = len(ARCH_ORDER)
    bar_width = 0.8 / n_archs
    x_base = np.arange(max_wl)

    for i, arch in enumerate(ARCH_ORDER):
        k_vals = [k for _, k in weight_layers_k[arch]]
        # Pad shorter architectures
        while len(k_vals) < max_wl:
            k_vals.append(0)
        x_pos = x_base + i * bar_width - (n_archs - 1) * bar_width / 2
        ax.bar(x_pos, k_vals, width=bar_width,
               color=colors[i], label=arch, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Weight layer index")
    ax.set_ylabel("Adaptive $k$ ($\\rho \\geq 0.9$)")
    ax.set_title("Subspace dimension per layer")
    # Show every Nth tick to avoid crowding
    tick_step = max(1, max_wl // 15)
    tick_pos = np.arange(0, max_wl, tick_step)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(j) for j in tick_pos], fontsize=8)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

    plt.tight_layout()
    out_fig = FIG_DIR / f"{fig_id}-eigenspectrum.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    logging.info(f"Figure saved to {out_fig}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="Run eigenvalue extraction")
    parser.add_argument("--plot", action="store_true", help="Generate figure from cached data")
    parser.add_argument("--gpu", type=int, default=1, help="GPU id for extraction")
    parser.add_argument("--dataset", type=str, default="cylinder",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to process")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.extract and not args.plot:
        args.extract = True
        args.plot = True

    if args.extract:
        all_data = extract_all(args.gpu, args.dataset)
    else:
        all_data = None

    if args.plot:
        if all_data is None:
            cache_path = CACHE_DIR / f"eigenspectrum_data_{args.dataset}.pt"
            if not cache_path.exists():
                logging.error(f"No cached data at {cache_path}. Run with --extract first.")
                return
            all_data = torch.load(cache_path, map_location="cpu", weights_only=False)
            logging.info(f"Loaded cached eigenvalue data from {cache_path}")
        plot_figure(all_data, args.dataset)


if __name__ == "__main__":
    main()
