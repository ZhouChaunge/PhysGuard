"""
Null-Space Fine-tuning Training Script for RealPDEBench.

Two-phase training:
  Phase 1: Load pre-trained model (trained on numerical/simulated data).
           Compute null-space projection from simulated data.
  Phase 2: Fine-tune on real data with gradient projection into the null space.

Usage:
    python -m realpdebench.train_nullspace --config configs/cylinder/fno_nullspace.yaml

    # Or equivalently:
    python -m realpdebench.train_nullspace \
        --config configs/cylinder/fno.yaml \
        --checkpoint_path /path/to/pretrained.pth \
        --train_data_type real \
        --ns_n_components 50 \
        --ns_alpha 1.0
"""

import os
import sys
import torch
import torch.nn as nn
import tqdm
import logging
import time
import datetime
import numpy as np
import argparse

from realpdebench.data.combustion_dataset import CombustionDataset
from realpdebench.data.combustion_hf_dataset import CombustionHFDataset
from realpdebench.data.fluid_dataset import FSI, Cylinder, ControlledCylinder, Foil
from realpdebench.data.fluid_hf_dataset import (
    CylinderHFDataset,
    FSIHFDataset,
    ControlledCylinderHFDataset,
    FoilHFDataset,
)
from realpdebench.data.data_normalizer import IdentityNormalizer, GaussianNormalizer, RangeNormalizer
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import set_seed, add_args_from_config, setup_logging, cycle
from realpdebench.utils.metrics import eval_metrics, mse_loss
from realpdebench.nullspace.null_space_projector import NullSpaceProjector
from realpdebench.nullspace.null_space_optimizer import NullSpaceOptimizer


def parse_gpu_ids(gpu_field):
    """Parse gpu field from YAML/CLI: int | list | comma-string → List[int]."""
    if isinstance(gpu_field, int):
        return [gpu_field]
    if isinstance(gpu_field, list):
        return [int(g) for g in gpu_field]
    if isinstance(gpu_field, str):
        # strip yaml list brackets if present
        gpu_field = gpu_field.strip().strip('[]')
        return [int(g.strip()) for g in gpu_field.split(',') if g.strip()]
    raise ValueError(f"Cannot parse gpu field: {gpu_field!r}")


class _TrainLossWrapper(nn.Module):
    """Expose model.train_loss() as forward() for DataParallel."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input, target):
        return self.model.train_loss(input, target)


parser = argparse.ArgumentParser(description="Null-Space Fine-tuning Configurations")
parser.add_argument("--config", type=str, default="configs/cylinder/fno_nullspace.yaml")
parser.add_argument("--gpu", default=0,
                    help="GPU id(s): single int (0), list ([0,1,2,3]) or comma string ('0,1,2,3')")
parser.add_argument("--exp_path", type=str, default=None,
                    help="Reuse an existing experiment directory (skip creating a new timestamp dir).")
parser.add_argument("--train_data_type", type=str, default="real",
                    help="Data type for fine-tuning. Should be 'real' for null-space fine-tuning.")
parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="Path to pre-trained checkpoint (trained on numerical data).")
parser.add_argument("--use_hf_dataset", action="store_true",
                    help="Use HuggingFace Arrow-backed dataset wrapper.")
parser.add_argument("--hf_auto_download", action="store_true",
                    help="Auto-download required HF artifacts if missing.")
parser.add_argument("--hf_repo_id", type=str, default="AI4Science-WestlakeU/RealPDEBench")
parser.add_argument("--hf_endpoint", type=str, default=None)
parser.add_argument("--hf_revision", type=str, default=None)

# Null-space specific arguments
parser.add_argument("--ns_n_components", type=int, default=200,
                    help="Max number of top eigenvectors (upper-bound cap when variance_threshold is set).")
parser.add_argument("--ns_variance_threshold", type=float, default=0.9,
                    help="Adaptive k: keep eigenvectors explaining this fraction of Fisher variance "
                         "(per layer). Set to 0 or negative to disable and use fixed ns_n_components.")
parser.add_argument("--ns_alpha", type=float, default=1.0,
                    help="Protection strength (1.0 = full null-space projection).")
parser.add_argument("--ns_progressive", action="store_true",
                    help="Enable progressive relaxation of protection.")
parser.add_argument("--ns_beta_min", type=float, default=0.3,
                    help="Minimum protection strength for progressive relaxation.")
parser.add_argument("--ns_layer_wise", type=bool, default=True,
                    help="Use per-layer null-space projection.")
parser.add_argument("--ns_max_samples", type=int, default=200,
                    help="Max samples for Fisher computation.")
parser.add_argument("--ns_projection_path", type=str, default=None,
                    help="Path to load/save pre-computed null-space projections.")
parser.add_argument("--ns_protected_layers", type=str, default=None,
                    help="Comma-separated list of parameter name substrings to protect. "
                         "e.g. 'spectral_convs' or 'spectral_convs,convs'. "
                         "If not set, ALL parameters are protected (original behaviour).")
parser.add_argument("--use_amp", type=bool, default=True,
                    help="Use BF16 autocast for training. Set to false for deep models "
                         "(e.g. DPOT) that are prone to numerical overflow with BF16.")
parser.add_argument("--start_iter", type=int, default=0,
                    help="Resume from this iteration (skip already-done steps). "
                         "The LR scheduler will be fast-forwarded accordingly.")


def build_datasets(args):
    """Build train (real), val (real), and normalizer (numerical) datasets."""
    hf_common_kwargs = {}
    if args.use_hf_dataset:
        hf_common_kwargs = {
            "hf_auto_download": bool(getattr(args, 'hf_auto_download', False)),
            "hf_repo_id": args.hf_repo_id,
            "hf_endpoint": args.hf_endpoint,
            "hf_revision": getattr(args, 'hf_revision', None),
        }

    dataset_map = {
        'combustion': (CombustionHFDataset if args.use_hf_dataset else CombustionDataset),
        'fsi': (FSIHFDataset if args.use_hf_dataset else FSI),
        'cylinder': (CylinderHFDataset if args.use_hf_dataset else Cylinder),
        'controlled_cylinder': (ControlledCylinderHFDataset if args.use_hf_dataset else ControlledCylinder),
        'foil': (FoilHFDataset if args.use_hf_dataset else Foil),
    }

    if args.dataset_name not in dataset_map:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    DatasetClass = dataset_map[args.dataset_name]
    logging.info(f"Using {DatasetClass.__name__}")

    # Train dataset: real data for fine-tuning
    train_dataset = DatasetClass(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode='train',
        dataset_type=args.train_data_type,
        mask_prob=args.mask_prob,
        noise_scale=args.noise_scale,
        **hf_common_kwargs,
    )

    # Val dataset: always real
    val_dataset = DatasetClass(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode='val',
        dataset_type='real',
        **hf_common_kwargs,
    )

    # Normalizer dataset: from numerical data (consistent with pre-training)
    normalizer_dataset = DatasetClass(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode='train',
        dataset_type='numerical',
        **hf_common_kwargs,
    )

    # Numerical dataset for Fisher computation (same distribution as pre-training)
    fisher_dataset = DatasetClass(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode='train',
        dataset_type='numerical',
        **hf_common_kwargs,
    )

    # Numerical validation set for forgetting tracking.
    # Try val split first; fall back to test (cylinder only has
    # test_index_numerical.json).
    # Guard: some scenarios (e.g. combustion/003) have 0 numerical val/test
    # samples because all numerical sims are allocated to training for Fisher
    # computation.  Return None so downstream code skips forgetting tracking.
    val_num_dataset = None
    for split in ('val', 'test'):
        try:
            _candidate = DatasetClass(
                dataset_name=args.dataset_name,
                dataset_root=args.dataset_root,
                mode=split,
                dataset_type='numerical',
                **hf_common_kwargs,
            )
            if len(_candidate) > 0:
                val_num_dataset = _candidate
                break
            else:
                logging.info(
                    f"Numerical {split} set exists but has 0 samples "
                    f"(dataset={args.dataset_name}); skipping."
                )
        except (FileNotFoundError, Exception):
            continue

    if val_num_dataset is None:
        logging.info(
            f"No numerical val/test data available for '{args.dataset_name}'. "
            f"Forgetting tracker will be disabled."
        )

    return train_dataset, val_dataset, normalizer_dataset, fisher_dataset, val_num_dataset


if __name__ == "__main__":
    args = parser.parse_args()
    # Resolve config path
    if not os.path.exists(args.config):
        candidate = os.path.join(os.path.dirname(__file__), args.config)
        if os.path.exists(candidate):
            args.config = candidate
    args = add_args_from_config(args, parser)
    gpu_ids = parse_gpu_ids(getattr(args, 'gpu', 0))
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    if getattr(args, 'exp_path', None):
        exp_path = args.exp_path
    else:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        exp_path = os.path.join(
            args.results_path, args.model_name,
            args.exp_name + '_nsft',
            current_time
        )
    os.makedirs(exp_path, exist_ok=True)

    writer = setup_logging(exp_path, args.is_use_tb)
    if args.is_use_tb:
        for key, value in vars(args).items():
            writer.add_text(key, str(value), 0)
    logging.info(f'args: {args}')

    # =========================================================================
    # Phase 0: Load datasets
    # =========================================================================
    train_dataset, val_dataset, normalizer_dataset, fisher_dataset, val_num_dataset = build_datasets(args)

    train_dataloader = cycle(torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers
    ))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size,
        shuffle=False, pin_memory=True, num_workers=args.num_workers
    )
    fisher_dataloader = torch.utils.data.DataLoader(
        fisher_dataset, batch_size=args.test_batch_size,
        shuffle=True, pin_memory=True, num_workers=min(args.num_workers, 4)
    )

    # Forgetting tracker: fixed 200-sample numerical subset
    # Guard: val_num_dataset is None when numerical val/test has 0 samples
    # (e.g. combustion dataset where all numerical data is used for Fisher).
    val_num_dataloader = None
    if val_num_dataset is not None and len(val_num_dataset) > 0:
        _num_val_size = min(200, len(val_num_dataset))
        val_num_subset = torch.utils.data.Subset(val_num_dataset, list(range(_num_val_size)))
        val_num_dataloader = torch.utils.data.DataLoader(
            val_num_subset, batch_size=args.test_batch_size,
            shuffle=False, pin_memory=True, num_workers=args.num_workers
        )

    logging.info(f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
                 f"Fisher: {len(fisher_dataset)}")

    # Setup data normalizer
    if args.normalizer == 'none':
        data_normalizer = IdentityNormalizer(device=device)
    elif args.normalizer == 'gaussian':
        data_normalizer = GaussianNormalizer(normalizer_dataset, device=device)
    elif args.normalizer == 'range':
        data_normalizer = RangeNormalizer(normalizer_dataset, device=device)
    else:
        raise ValueError(f"Normalizer {args.normalizer} not supported")

    # =========================================================================
    # Phase 1: Load pre-trained model and compute null-space projection
    # =========================================================================
    model = load_model(train_dataset, device=device, **vars(args))
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters: {num_params}")

    # Load pre-trained checkpoint (mandatory for null-space fine-tuning)
    assert args.checkpoint_path is not None, \
        "checkpoint_path is required for null-space fine-tuning. " \
        "Provide a model pre-trained on numerical data."
    # Resolve relative checkpoint/projection paths against exp_path when resuming
    if args.exp_path and not os.path.isabs(args.checkpoint_path):
        args.checkpoint_path = os.path.join(args.exp_path, args.checkpoint_path)
    ns_proj_path = getattr(args, 'ns_projection_path', None)
    if args.exp_path and ns_proj_path and not os.path.isabs(ns_proj_path):
        args.ns_projection_path = os.path.join(args.exp_path, ns_proj_path)
    meta_data = model.load_checkpoint(args.checkpoint_path, device)
    logging.info(f"Pre-trained checkpoint loaded from {args.checkpoint_path}")

    # Compute or load null-space projections
    ns_n_components = getattr(args, 'ns_n_components', 200)
    ns_alpha = getattr(args, 'ns_alpha', 1.0)
    ns_progressive = getattr(args, 'ns_progressive', False)
    ns_beta_min = getattr(args, 'ns_beta_min', 0.3)
    ns_layer_wise = getattr(args, 'ns_layer_wise', True)
    ns_max_samples = getattr(args, 'ns_max_samples', 200)
    ns_projection_path = getattr(args, 'ns_projection_path', None)
    _vt = getattr(args, 'ns_variance_threshold', 0.9)
    ns_variance_threshold = _vt if (_vt is not None and _vt > 0) else None

    # Parse protected_layers: comma-separated string -> list, or None for all
    _protected_raw = getattr(args, 'ns_protected_layers', None)
    if _protected_raw:
        ns_protected_layers = [s.strip() for s in _protected_raw.split(',') if s.strip()]
    else:
        ns_protected_layers = None

    projector = NullSpaceProjector(
        n_components=ns_n_components,
        alpha=ns_alpha,
        progressive=ns_progressive,
        total_steps=args.num_update,
        beta_min=ns_beta_min,
        layer_wise=ns_layer_wise,
        device=str(device),
        protected_layers=ns_protected_layers,
        variance_threshold=ns_variance_threshold,
    )

    if ns_projection_path and os.path.exists(ns_projection_path):
        projector.load(ns_projection_path)
        logging.info(f"Loaded pre-computed null-space projections from {ns_projection_path}")
    else:
        logging.info("Computing null-space projections from numerical data...")
        projector.compute_projection(
            model=model,
            dataloader=fisher_dataloader,
            data_normalizer=data_normalizer,
            max_samples=ns_max_samples,
        )
        # Save projections for reuse
        proj_save_path = ns_projection_path or os.path.join(exp_path, "null_space_projections.pt")
        projector.save(proj_save_path)

    # =========================================================================
    # Phase 2: Fine-tune with null-space constrained optimizer
    # =========================================================================
    base_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = NullSpaceOptimizer(base_optimizer, model, projector)

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=args.step_size, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=args.num_update)
    else:
        raise ValueError(f"Scheduler {args.scheduler} not supported")

    # Fast-forward scheduler to resume position
    start_iter = getattr(args, 'start_iter', 0)
    if start_iter > 0:
        logging.info(f"Resuming from iter {start_iter}, fast-forwarding scheduler by {start_iter} steps.")
        for _ in range(start_iter):
            scheduler.step()

    start_time = time.time()
    best_iteration = 0

    # Wrap model with DataParallel for multi-GPU training.
    # Projector & optimizer always operate on the raw `model` object.
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        dp_train = nn.DataParallel(_TrainLossWrapper(model), device_ids=gpu_ids)
        logging.info(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        dp_train = _TrainLossWrapper(model)

    logging.info(f"Start null-space fine-tuning on {device} "
                 f"(alpha={ns_alpha}, n_components(cap)={ns_n_components}, "
                 f"variance_threshold={ns_variance_threshold}, "
                 f"progressive={ns_progressive})")

    pbar = tqdm.tqdm(range(start_iter + 1, args.num_update + 1))
    best_val_loss = float('inf')
    total_loss = 0.
    count = 0

    all_train_losses = []
    all_val_losses = {
        'normalized_mse': [], 'rmse': [], 'mae': [], 'rel_l2_error': [],
        'r2': [], 'ke_error': [], 'f_error': [], 'low_f_error': [],
        'mid_f_error': [], 'high_f_error': [],
        'rel_low_f_error': [], 'rel_mid_f_error': [], 'rel_high_f_error': [],
        'freq_error': [],
    }

    for iteration in pbar:
        model.train()

        input, target = next(train_dataloader)
        optimizer.zero_grad()
        input, target = data_normalizer.preprocess(input, target)

        # BF16 autocast: avoids FP16 overflow for large Transformer models.
        # Disable (use_amp: false in yaml) for very deep models like DPOT whose
        # 48-layer AFNO stack accumulates enough BF16 rounding error to produce inf.
        use_amp = getattr(args, 'use_amp', True)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = dp_train(input, target).mean()
        else:
            loss = dp_train(input, target).mean()

        if not torch.isfinite(loss):
            logging.warning(f"Non-finite loss ({loss.item()}) at iter {iteration}, skipping step.")
            scheduler.step()
            count += 1
            pbar.set_postfix(loss=loss.item(), beta=f"{projector.get_protection_strength():.2f}")
            continue

        loss.backward()

        if args.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        projector.project_gradients(model)

        base_optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        count += 1
        pbar.set_postfix(
            loss=loss.item(),
            beta=f"{projector.get_protection_strength():.2f}"
        )

        all_train_losses.append(loss.item())

        if args.is_use_tb:
            writer.add_scalar("train_loss", loss.item(), iteration)
            writer.add_scalar("protection_strength", projector.get_protection_strength(), iteration)

        if iteration % int(args.num_update / 50) == 0:
            model.eval()
            normalized_val_loss = 0.

            pred_list, target_list = [], []
            with torch.no_grad():
                for input, target in val_dataloader:
                    b = input.size(0)
                    if 'unmeasured_c' not in locals():
                        unmeasured_c = 0
                        for c_ in range(target.shape[-1]):
                            if torch.all(target[..., c_] == 0):
                                unmeasured_c += 1
                    c = target.shape[-1] - unmeasured_c

                    input, target = data_normalizer.preprocess(input, target)

                    pred = model(input)
                    normalized_val_loss += mse_loss(
                        pred[..., :c], target[..., :c]
                    ).reshape(b, -1).mean().item()

                    _, pred = data_normalizer.postprocess(input, pred)
                    _, target = data_normalizer.postprocess(input, target)

                    pred_list.append(pred.cpu())
                    target_list.append(target.cpu())

                normalized_val_loss /= len(val_dataloader)
                (val_rmse, val_mae, val_rel_l2_error, val_r2, val_ke_error,
                 val_f_error, val_low_f_error, val_mid_f_error,
                 val_high_f_error, val_rel_low_f_error, val_rel_mid_f_error,
                 val_rel_high_f_error, val_freq_error) = \
                    eval_metrics(
                        torch.cat(pred_list, dim=0),
                        torch.cat(target_list, dim=0), c
                    )

                all_val_losses['normalized_mse'].append(normalized_val_loss)
                all_val_losses['rmse'].append(val_rmse)
                all_val_losses['mae'].append(val_mae)
                all_val_losses['rel_l2_error'].append(val_rel_l2_error)
                all_val_losses['r2'].append(val_r2)
                all_val_losses['ke_error'].append(val_ke_error)
                all_val_losses['f_error'].append(val_f_error)
                all_val_losses['low_f_error'].append(val_low_f_error)
                all_val_losses['mid_f_error'].append(val_mid_f_error)
                all_val_losses['high_f_error'].append(val_high_f_error)
                all_val_losses['rel_low_f_error'].append(val_rel_low_f_error)
                all_val_losses['rel_mid_f_error'].append(val_rel_mid_f_error)
                all_val_losses['rel_high_f_error'].append(val_rel_high_f_error)
                all_val_losses['freq_error'].append(val_freq_error)

                if val_rmse < best_val_loss:
                    best_iteration = iteration
                    best_val_loss = val_rmse

            # ── Forgetting tracker: numerical validation ───────────────────────
            num_rmse_str = ''
            if val_num_dataloader is not None:
                pred_list_num, target_list_num = [], []
                unmeasured_c_num = 0
                with torch.no_grad():
                    for inp_n, tgt_n in val_num_dataloader:
                        b_n = inp_n.size(0)
                        if not pred_list_num:
                            unmeasured_c_num = sum(
                                1 for c_ in range(tgt_n.shape[-1])
                                if torch.all(tgt_n[..., c_] == 0)
                            )
                        inp_n, tgt_n = data_normalizer.preprocess(inp_n, tgt_n)
                        pred_n = model(inp_n)
                        c_n = tgt_n.shape[-1] - unmeasured_c_num
                        _, pred_n  = data_normalizer.postprocess(inp_n, pred_n)
                        _, tgt_n   = data_normalizer.postprocess(inp_n, tgt_n)
                        pred_list_num.append(pred_n.cpu())
                        target_list_num.append(tgt_n.cpu())
                if pred_list_num:
                    pred_num_all   = torch.cat(pred_list_num,   dim=0)
                    target_num_all = torch.cat(target_list_num, dim=0)
                    metrics_num    = eval_metrics(pred_num_all, target_num_all, c_n)
                    num_rmse       = metrics_num[0]
                    all_val_losses.setdefault('num_rmse', []).append(num_rmse)
                    num_rmse_str   = f'  num_rmse={num_rmse:.5f} [forgetting]'
                    if args.is_use_tb:
                        writer.add_scalar("num_rmse", num_rmse, iteration)

            logging.info(
                f"\nIteration {iteration}, train loss: {total_loss / count:.5f}, "
                f"protection_beta: {projector.get_protection_strength():.3f}"
            )
            logging.info(
                f"Validation results: \n"
                f"normalized mse loss: {normalized_val_loss:.5f}, "
                f"rmse: {val_rmse:.5f}, mae: {val_mae:.5f}, "
                f"rel l2 error: {val_rel_l2_error:.5f}, "
                f"r2: {val_r2:.5f}, ke error: {val_ke_error:.5f}, "
                f"f error: {val_f_error:.5f}, "
                f"low f error: {val_low_f_error:.5f}, "
                f"mid f error: {val_mid_f_error:.5f}, "
                f"high f error: {val_high_f_error:.5f}, "
                f"rel low f error: {val_rel_low_f_error:.5f}, "
                f"rel mid f error: {val_rel_mid_f_error:.5f}, "
                f"rel high f error: {val_rel_high_f_error:.5f}, "
                f"freq error: {val_freq_error:.5f}"
                f"{num_rmse_str}"
            )
            total_loss = 0.
            count = 0

            if args.is_use_tb:
                writer.add_scalar("normalized_val_loss", normalized_val_loss, iteration)
                writer.add_scalar("val_rmse", val_rmse, iteration)
                writer.add_scalar("val_mae", val_mae, iteration)
                writer.add_scalar("val_rel_l2_error", val_rel_l2_error, iteration)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'train_losses': all_train_losses,
                'val_losses': all_val_losses,
                'iteration': iteration,
                'best_iteration': best_iteration,
                'best_val_loss': best_val_loss,
                'ns_config': {
                    'n_components': ns_n_components,
                    'alpha': ns_alpha,
                    'progressive': ns_progressive,
                    'beta_min': ns_beta_min,
                },
            }
            torch.save(checkpoint, os.path.join(exp_path, f"model_{iteration:04d}.pth"))

    end_time = time.time()
    logging.info(
        f"Null-space fine-tuning complete. "
        f"Best iteration: {best_iteration}, "
        f"Best val RMSE: {best_val_loss:.5f}, "
        f"Time: {(end_time - start_time) / 60:.2f} min"
    )
    logging.info(f"Results saved at {exp_path}")

    if args.is_use_tb:
        writer.close()
