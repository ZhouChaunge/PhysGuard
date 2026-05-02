"""
physguard._engine  —  Shared training engine for PhysGuard experiments.
==========================================================================
This module is *not* meant to be invoked directly. Use the public entry
points which select the running mode for you:

    python -m physguard.pretrain           --config configs/.../1_pretrain.yaml
    python -m physguard.finetune_baselines --config configs/.../2_dft.yaml

The 'gpu' field in the yaml controls device placement:
  gpu: 0          →  single GPU  cuda:0
  gpu: [0, 1]     →  2-GPU  DDP (auto-spawns subprocesses)
  gpu: [0,1,2,3]  →  4-GPU  DDP (auto-spawns subprocesses)

No manual torchrun needed.
"""

import os
import sys
import inspect
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import tqdm
import logging
import time
import datetime
import contextlib
import numpy as np
import argparse

from benchmark.data.combustion_hf_dataset import CombustionHFDataset
from benchmark.data.combustion_hf_dataset import CombustionHFDataset as CombustionDataset  # back-compat alias
from benchmark.data.fluid_hf_dataset import (
    CylinderHFDataset,
    FSIHFDataset,
    ControlledCylinderHFDataset,
    FoilHFDataset,
    CylinderHFDataset as Cylinder,                # back-compat aliases
    ControlledCylinderHFDataset as ControlledCylinder,
    FSIHFDataset as FSI,
    FoilHFDataset as Foil,
)
from benchmark.data.data_normalizer import (
    IdentityNormalizer, GaussianNormalizer, RangeNormalizer
)
from benchmark.model.load_model import load_model
from benchmark.utils.utils import (
    set_seed, add_args_from_config, setup_logging, cycle,
    banner, log_blank, log_header_block, log_run_config, table_header_lines, format_metric_row,
    CSVMetricWriter, VAL_METRIC_KEYS,
)
from benchmark.utils.metrics import eval_metrics, mse_loss


# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse 'gpu' field from yaml
# ─────────────────────────────────────────────────────────────────────────────

def parse_gpu_ids(gpu_field):
    """
    Supports three yaml formats:
        gpu: 0          → [0]       single GPU
        gpu: [0, 1]     → [0, 1]   2-GPU
        gpu: 0,1,2,3    → [0,1,2,3] 4-GPU (comma-separated string)
    Returns: List[int]
    """
    if isinstance(gpu_field, int):
        return [gpu_field]
    if isinstance(gpu_field, list):
        return [int(g) for g in gpu_field]
    if isinstance(gpu_field, str):
        return [int(g.strip()) for g in gpu_field.split(',')]
    raise ValueError(f"Cannot parse gpu field: {gpu_field!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build datasets
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(args):
    name = args.dataset_name
    hf_kwargs = {}
    if args.use_hf_dataset:
        hf_kwargs = {
            "hf_auto_download": bool(args.hf_auto_download),
            "hf_repo_id":       args.hf_repo_id,
            "hf_endpoint":      args.hf_endpoint,
            "hf_revision":      args.hf_revision,
        }

    dataset_map = {
        'combustion':          (CombustionDataset,  CombustionHFDataset),
        'fsi':                 (FSI,                FSIHFDataset),
        'cylinder':            (Cylinder,           CylinderHFDataset),
        'controlled_cylinder': (ControlledCylinder, ControlledCylinderHFDataset),
        'foil':                (Foil,               FoilHFDataset),
    }
    if name not in dataset_map:
        raise ValueError(f"Dataset {name} not supported")

    LocalCls, HFCls = dataset_map[name]
    DatasetClass = HFCls if args.use_hf_dataset else LocalCls

    common = dict(dataset_name=name, dataset_root=args.dataset_root)
    train_dataset = DatasetClass(**common, mode='train',
                                  dataset_type=args.train_data_type,
                                  mask_prob=args.mask_prob,
                                  noise_scale=args.noise_scale, **hf_kwargs)
    val_dataset   = DatasetClass(**common, mode='val',
                                  dataset_type='real', **hf_kwargs)
    norm_dataset  = DatasetClass(**common, mode='train',
                                  dataset_type='numerical', **hf_kwargs)

    # Numerical validation set for forgetting tracking during fine-tuning.
    # Try val split first (combustion has val_index_numerical.json);
    # fall back to test split (cylinder only has test_index_numerical.json).
    val_num_dataset = None
    for split in ('val', 'test'):
        try:
            val_num_dataset = DatasetClass(**common, mode=split,
                                           dataset_type='numerical', **hf_kwargs)
            break
        except (FileNotFoundError, Exception):
            continue

    return train_dataset, val_dataset, norm_dataset, val_num_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Core training function (used for both single-GPU and each DDP worker)
# ─────────────────────────────────────────────────────────────────────────────

def train(rank, gpu_ids, args, exp_path):
    """
    rank      : index of current process within gpu_ids (0-based)
    gpu_ids   : list of all GPU ids participating in training, e.g. [0, 1]
    args      : parsed arguments
    exp_path  : experiment output directory (already created by main process)
    """
    world_size = len(gpu_ids)
    use_ddp    = world_size > 1
    is_main    = rank == 0
    local_gpu  = gpu_ids[rank]

    # ── Initialize process group ───────────────────────────────────────────────
    if use_ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.ddp_port)   # configurable in yaml, default 12355
        dist.init_process_group(
            backend    = 'nccl',
            rank       = rank,
            world_size = world_size,
        )
        torch.cuda.set_device(local_gpu)

    device = torch.device(f'cuda:{local_gpu}' if torch.cuda.is_available() else 'cpu')

    # ── Seed (different per process to avoid identical augmentation) ─────────
    set_seed(args.seed + rank)

    # ── Logging (rank-0 only) ────────────────────────────────────────────────
    writer = None
    if is_main:
        writer = setup_logging(exp_path, args.is_use_tb)
        if args.is_use_tb:
            for key, value in vars(args).items():
                writer.add_text(key, str(value), 0)
        gpu_str = (f"DDP \u00d7 {world_size} GPUs {gpu_ids}" if use_ddp
                   else f"single GPU {local_gpu}")
        # Override args.gpu display so Run config 'gpu' row shows the actual mode
        # (was redundant with a separate gpu_mode line above).
        try:
            args.gpu = gpu_str  # purely cosmetic; Run config dumps via vars(args)
        except Exception:
            pass
        title = "PhysGuard \u00b7 " + ("Finetune" if args.is_finetune else "Pretrain")
        log_header_block(title, exp_path)
        log_run_config(args)
        log_blank()
        logging.info(banner("Stage 1/3 \u00b7 Loading data"))

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dataset, val_dataset, normalizer_dataset, val_num_dataset = build_datasets(args)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                            rank=rank, shuffle=True, drop_last=True)
    else:
        train_sampler = None

    train_dataloader = cycle(
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size = args.train_batch_size,   # per-GPU batch size
            sampler    = train_sampler,
            shuffle    = (train_sampler is None),
            pin_memory = True,
            num_workers= args.num_workers,
        )
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size  = args.test_batch_size,
        shuffle     = False,
        pin_memory  = True,
        num_workers = args.num_workers,
    )

    # Forgetting tracker: fixed 200-sample numerical subset (rank-0 only)
    val_num_dataloader = None
    if args.is_finetune and val_num_dataset is not None:
        _num_val_size = min(200, len(val_num_dataset))
        val_num_subset = torch.utils.data.Subset(val_num_dataset, list(range(_num_val_size)))
        val_num_dataloader = torch.utils.data.DataLoader(
            val_num_subset,
            batch_size  = args.test_batch_size,
            shuffle     = False,
            pin_memory  = True,
            num_workers = args.num_workers,
        )

    if is_main:
        logging.info(f"[data] all splits ready at: {train_dataset.dataset_path}")
        if use_ddp:
            logging.info(
                f"[data] global train batch size = {args.train_batch_size} \u00d7 {world_size} = "
                f"{args.train_batch_size * world_size}"
            )
        log_blank()
        logging.info(banner("Stage 2/3 \u00b7 Building model"))

    # ── Normalizer ────────────────────────────────────────────────────────────
    # Stats cache (mean_std.pt / max.pt) is pre-computed by the main process
    # before mp.spawn, so loading here is always a fast cache hit.
    if args.normalizer == 'none':
        data_normalizer = IdentityNormalizer(device=device)
    elif args.normalizer == 'gaussian':
        data_normalizer = GaussianNormalizer(normalizer_dataset, device=device)
    elif args.normalizer == 'range':
        data_normalizer = RangeNormalizer(normalizer_dataset, device=device)
    else:
        raise ValueError(f"Normalizer {args.normalizer} not supported")

    # ── Model ────────────────────────────────────────────────────────────────
    model = load_model(train_dataset, device=device, **vars(args))
    if is_main:
        # Compact one-line summary: '[model] fno   in=[20, 64, 128, 3]   out=[20, 64, 128, 3]'
        _in_shape  = list(train_dataset[0][0].shape)
        _out_shape = list(train_dataset[0][1].shape)
        logging.info(f"[model] {args.model_name}   in={_in_shape}   out={_out_shape}")
        num_params = sum(p.numel() for p in model.parameters())
        _mib = num_params * 4 / (1024 * 1024)
        logging.info(f"[model] params: {num_params:,}   (\u2248 {_mib:.1f} MiB fp32)")

    if use_ddp:
        # CNO and DPOT have unused parameters in certain configurations
        # (e.g. CNO invariant blocks, DPOT n_cls embedding not used in all datasets)
        find_unused = (args.model_name in ('cno', 'dpot'))
        model = DDP(model, device_ids=[local_gpu], output_device=local_gpu,
                    find_unused_parameters=find_unused)

    # raw_model: unwrapped model from DDP, used for forward/custom methods/saving weights
    raw_model = model.module if use_ddp else model

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_update)
    else:
        raise ValueError(f"Scheduler {args.scheduler} not supported")

    # ── Finetune ──────────────────────────────────────────────────────────────
    if args.is_finetune:
        raw_model.load_checkpoint(args.checkpoint_path, device)
        if is_main:
            logging.info(f"Checkpoint loaded from: {args.checkpoint_path}")

    # Detect diffusion model (WDNO) early — needed by both the EWC Fisher loop and the main training loop
    _use_diffusion_loop = 'target' in inspect.signature(raw_model.forward).parameters

    # ── Regularization setup (L2-FT / EWC-FT) ────────────────────────────────
    reg_type   = getattr(args, 'reg_type',   'none')
    reg_lambda = getattr(args, 'reg_lambda', 0.0)

    # Save a frozen copy of pretrained weights for L2 / EWC anchor
    pretrained_params = None
    fisher_diag       = None
    if args.is_finetune and reg_type in ('l2', 'ewc') and reg_lambda > 0:
        pretrained_params = {
            n: p.detach().clone()
            for n, p in raw_model.named_parameters()
            if p.requires_grad
        }
        if reg_type == 'ewc':
            # Compute diagonal Fisher information ≈ E[|∇L|²] over ewc_num_samples.
            # Fisher must be real-valued; complex params (e.g. FNO SpectralConv) require
            # squaring magnitude (real²+imag²) rather than squaring the complex value.
            ewc_num_samples = getattr(args, 'ewc_num_samples', 200)
            raw_model.train()
            # Initialize accumulators as real tensors regardless of parameter dtype
            fisher_acc = {
                n: torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                   if p.is_complex()
                   else torch.zeros_like(p)
                for n, p in pretrained_params.items()
            }
            n_seen = 0
            # Use NUMERICAL (simulated) data to estimate Fisher — this matches the
            # pre-training distribution and captures which parameters encode numerical
            # physics knowledge that should be preserved during sim-to-real transfer.
            # (Using real data here would protect parameters important for real data,
            #  which is the task being fine-tuned on — completely backwards.)
            ewc_iter = iter(torch.utils.data.DataLoader(
                normalizer_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            ))
            # DDP fix: call backward through raw_model (bypassing DDP wrapper) with
            # no_sync() to prevent the DDP reducer from waiting on unused-parameter
            # buckets (find_unused_parameters=True for CNO/DPOT) which would cause
            # a hang.  After the loop we manually all-reduce the per-rank Fisher
            # accumulators to produce the averaged Fisher across all GPUs.
            _no_sync_ctx = model.no_sync if use_ddp else contextlib.nullcontext
            while n_seen < ewc_num_samples:
                try:
                    inp_e, tgt_e = next(ewc_iter)
                except StopIteration:
                    break
                inp_e, tgt_e = data_normalizer.preprocess(inp_e, tgt_e)
                raw_model.zero_grad()
                with _no_sync_ctx():
                    if _use_diffusion_loop:
                        loss_e = raw_model(inp_e, tgt_e).mean()
                    else:
                        pred_e = raw_model(inp_e)
                        loss_e = mse_loss(pred_e, tgt_e).mean()
                    loss_e.backward()
                for n, p in raw_model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        g = p.grad.detach()
                        # For complex grads compute |g|² = real²+imag² (always real)
                        fisher_acc[n] += (g.real ** 2 + g.imag ** 2) if g.is_complex() else g ** 2
                n_seen += inp_e.size(0)
            # Aggregate Fisher estimates across DDP ranks (each rank processed
            # n_seen samples independently; all-reduce sums them, then we divide
            # by world_size * n_seen to get the global average).
            if use_ddp:
                for n in fisher_acc:
                    dist.all_reduce(fisher_acc[n], op=dist.ReduceOp.SUM)
                scale = max(n_seen * world_size, 1)
            else:
                scale = max(n_seen, 1)
            fisher_diag = {n: fisher_acc[n] / scale for n in fisher_acc}
            raw_model.zero_grad()
            if is_main:
                logging.info(
                    f"EWC: computed diagonal Fisher over {n_seen} samples "
                    f"(ewc_num_samples={ewc_num_samples})"
                )

    # ── Training state ───────────────────────────────────────────────────────
    start_time     = time.time()
    best_iteration = 0
    best_val_loss  = float('inf')
    total_loss     = 0.0
    count          = 0
    unmeasured_c   = None
    start_iter     = 1

    all_train_losses = []
    all_val_losses   = {k: [] for k in [
        'normalized_mse', 'rmse', 'mae', 'rel_l2_error', 'r2',
        'ke_error', 'f_error', 'low_f_error', 'mid_f_error', 'high_f_error',
        'rel_low_f_error', 'rel_mid_f_error', 'rel_high_f_error', 'freq_error',
    ]}

    # ── Resume from checkpoint ────────────────────────────────────────────────
    resume_path = getattr(args, 'resume_checkpoint', None)
    if resume_path and os.path.exists(resume_path):
        if is_main:
            logging.info(f"[Resume] Restoring from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        start_iter     = ckpt['iteration'] + 1
        best_iteration = ckpt.get('best_iteration', 0)
        best_val_loss  = ckpt.get('best_val_loss', float('inf'))
        all_train_losses = ckpt.get('train_losses', [])
        all_val_losses   = ckpt.get('val_losses', all_val_losses)
        # Fast-forward scheduler to resume step
        for _ in range(ckpt['iteration']):
            scheduler.step()
        if is_main:
            logging.info(f"[Resume] Resuming from iteration {start_iter}, best_val_loss={best_val_loss:.5f}")

    if is_main:
        log_blank()
        logging.info(banner(f"Stage 3/3 \u00b7 Training ({args.num_update} steps)"))
        for line in table_header_lines():
            logging.info(line)

    # Per-iter CSV: full-precision dump for downstream plotting
    csv_writer = CSVMetricWriter(os.path.join(exp_path, "metrics.csv")) if is_main else None

    pbar = tqdm.tqdm(range(start_iter, args.num_update + 1), disable=(not is_main))

    # ═════════════════════════════════════════════════════════════════════════
    # Main training loop
    # ═════════════════════════════════════════════════════════════════════════
    for iteration in pbar:
        _t_iter_start = time.time()
        model.train()

        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(iteration)

        input, target = next(train_dataloader)
        optimizer.zero_grad()
        input, target = data_normalizer.preprocess(input, target)

        # WDNO (diffusion model): forward accepts target in training mode and returns loss directly;
        # other models (FNO / DeepONet / DPOT, etc.) follow the standard pred→mse_loss path.
        if _use_diffusion_loop:
            loss = model(input, target).mean()  # Forward through DDP wrapper for gradient sync
        else:
            pred = model(input)  # Forward through DDP wrapper for gradient sync
            loss = mse_loss(pred, target).mean()

        # Record pure MSE before adding regularization penalty (for unbiased logging).
        mse_val = loss.item()

        # ── Regularization penalty (L2-FT / EWC-FT) ──────────────────────────
        if pretrained_params is not None and reg_lambda > 0:
            reg_loss = torch.tensor(0.0, device=device)
            for n, p in raw_model.named_parameters():
                if n in pretrained_params:
                    diff = p - pretrained_params[n]
                    # For complex params (e.g. FNO), use |diff|² = real²+imag² to keep reg_loss real.
                    # diff**2 on a complex tensor gives a complex result → loss becomes complex → backward() crashes.
                    sq = (diff.real ** 2 + diff.imag ** 2) if diff.is_complex() else diff ** 2
                    if reg_type == 'ewc' and fisher_diag is not None:
                        reg_loss = reg_loss + (fisher_diag[n] * sq).sum()
                    else:  # l2
                        reg_loss = reg_loss + sq.sum()
            loss = loss + 0.5 * reg_lambda * reg_loss

        loss.backward()

        if args.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        optimizer.step()
        scheduler.step()
        # Use pure MSE for logging so train loss curves remain comparable across run types
        # (dft / ewcft / l2ft).  The backward() above already used the full loss with penalty.
        total_loss += mse_val
        count += 1

        if is_main:
            pbar.set_postfix(loss=f"{mse_val:.5f}")
            all_train_losses.append(mse_val)
            if args.is_use_tb:
                writer.add_scalar("train_loss", mse_val, iteration)

        # ── Validation (rank-0 only, full evaluation; other ranks skip) ───────
        if iteration % max(1, int(args.num_update / 50)) == 0:
            if is_main:
                model.eval()
                normalized_val_loss = 0.0
                pred_list, target_list = [], []
                _max_val_batches = int(getattr(args, 'max_val_batches', 0) or 0)

                with torch.no_grad():
                    for _vi, (input, target) in enumerate(val_dataloader):
                        if _max_val_batches and _vi >= _max_val_batches:
                            break
                        b = input.size(0)

                        if unmeasured_c is None:
                            unmeasured_c = sum(
                                1 for c_ in range(target.shape[-1])
                                if torch.all(target[..., c_] == 0)
                            )
                        c = target.shape[-1] - unmeasured_c

                        input, target = data_normalizer.preprocess(input, target)

                        if _use_diffusion_loop:
                            # WDNO: use diffusion loss as validation metric instead of expensive sampling
                            model.eval()
                            val_loss_batch = raw_model.train_loss(input, target).mean().item()
                            normalized_val_loss += val_loss_batch
                            # cannot obtain pred, skip per-sample metrics
                            pred_list.append(target.cpu())   # placeholder
                            target_list.append(target.cpu())
                        else:
                            pred = raw_model(input)
                            normalized_val_loss += (
                                mse_loss(pred[..., :c], target[..., :c])
                                .reshape(b, -1).mean().item()
                            )

                            _, pred   = data_normalizer.postprocess(input, pred)
                            _, target = data_normalizer.postprocess(input, target)
                            pred_list.append(pred.cpu())
                            target_list.append(target.cpu())

                _val_n_batches = len(pred_list) if pred_list else len(val_dataloader)
                normalized_val_loss /= max(1, _val_n_batches)
                pred_all   = torch.cat(pred_list,   dim=0)
                target_all = torch.cat(target_list, dim=0)

                metrics = eval_metrics(pred_all, target_all, c)
                (val_rmse, val_mae, val_rel_l2_error, val_r2, val_ke_error,
                 val_f_error, val_low_f_error, val_mid_f_error, val_high_f_error,
                 val_rel_low_f_error, val_rel_mid_f_error, val_rel_high_f_error,
                 val_freq_error) = metrics

                for key, val in zip(all_val_losses.keys(), [normalized_val_loss] + list(metrics)):
                    all_val_losses[key].append(val)

                _is_new_best = val_rmse < best_val_loss
                if _is_new_best:
                    best_iteration = iteration
                    best_val_loss  = val_rmse

                # ── Forgetting tracker: numerical validation ───────────────────
                # Evaluate on a fixed numerical subset to quantify catastrophic
                # forgetting during fine-tuning.  A rising num_rmse alongside
                # falling val_rmse (real) is the signature of physical forgetting.
                num_rmse_str = ''
                if val_num_dataloader is not None and not _use_diffusion_loop:
                    pred_list_num, target_list_num = [], []
                    unmeasured_c_num = None
                    with torch.no_grad():
                        for _vi_n, (inp_n, tgt_n) in enumerate(val_num_dataloader):
                            if _max_val_batches and _vi_n >= _max_val_batches:
                                break
                            b_n = inp_n.size(0)
                            if unmeasured_c_num is None:
                                unmeasured_c_num = sum(
                                    1 for c_ in range(tgt_n.shape[-1])
                                    if torch.all(tgt_n[..., c_] == 0)
                                )
                            inp_n, tgt_n = data_normalizer.preprocess(inp_n, tgt_n)
                            pred_n = raw_model(inp_n)
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

                # _is_new_best was set above when comparing val_rmse against best_val_loss
                _metrics_row = {
                    'normalized_mse': normalized_val_loss,
                    'rmse':           val_rmse,
                    'mae':            val_mae,
                    'rel_l2':         val_rel_l2_error,
                    'r2':             val_r2,
                    'ke_err':         val_ke_error,
                    'f_err':          val_f_error,
                    'low_f':          val_low_f_error,
                    'mid_f':          val_mid_f_error,
                    'high_f':         val_high_f_error,
                    'rel_low':        val_rel_low_f_error,
                    'rel_mid':        val_rel_mid_f_error,
                    'rel_high':       val_rel_high_f_error,
                    'freq_err':       val_freq_error,
                }
                _lr_now = optimizer.param_groups[0]['lr']
                _dt_iter = time.time() - _t_iter_start
                _train_loss_avg = total_loss / count
                log_blank()
                logging.info(
                    format_metric_row(
                        iteration, args.num_update, _train_loss_avg,
                        _metrics_row, _lr_now, _dt_iter, _is_new_best,
                    )
                    + (f"   [num_rmse={all_val_losses['num_rmse'][-1]:.5f}]" if num_rmse_str else "")
                )

                if csv_writer is not None:
                    _csv_row = {'iter': iteration, 'train_loss': _train_loss_avg}
                    _csv_row.update(_metrics_row)
                    _csv_row['lr'] = _lr_now
                    _csv_row['dt_seconds'] = _dt_iter
                    _csv_row['is_best'] = int(_is_new_best)
                    if num_rmse_str:
                        _csv_row['num_rmse'] = all_val_losses['num_rmse'][-1]
                    csv_writer.write(_csv_row)
                total_loss = 0.0
                count      = 0

                if args.is_use_tb:
                    writer.add_scalar("normalized_val_loss", normalized_val_loss, iteration)
                    writer.add_scalar("val_rmse",            val_rmse,            iteration)
                    writer.add_scalar("val_mae",             val_mae,             iteration)
                    writer.add_scalar("val_rel_l2_error",    val_rel_l2_error,    iteration)

                # Use raw_model.state_dict() to avoid saving keys with "module." prefix
                checkpoint = {
                    'model_state_dict': raw_model.state_dict(),
                    'train_losses':     all_train_losses,
                    'val_losses':       all_val_losses,
                    'iteration':        iteration,
                    'best_iteration':   best_iteration,
                    'best_val_loss':    best_val_loss,
                    'gpu_ids':          gpu_ids,        # save multi-GPU config for debugging
                }
                torch.save(checkpoint,
                           os.path.join(exp_path, f"model_{iteration:04d}.pth"))

    # ── Cleanup ──────────────────────────────────────────────────────────────
    end_time = time.time()
    if is_main:
        if csv_writer is not None:
            csv_writer.close()
        logging.info("")
        logging.info(banner("Done"))
        logging.info(f"    best_iter     = {best_iteration}")
        logging.info(f"    best_val_rmse = {best_val_loss:.5f}")
        logging.info(f"    elapsed       = {(end_time - start_time)/60:.2f} min")
        logging.info(f"    artifacts     : training.log, metrics.csv, model_*.pth")
        logging.info(f"    saved at      : {exp_path}")
        logging.info("\u2550" * 76)
        if args.is_use_tb and writer:
            writer.close()

    if use_ddp:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Public launcher
# ─────────────────────────────────────────────────────────────────────────────

def launch(mode: str = "pretrain") -> None:
    """Parse CLI / YAML and run the training loop.

    This is the single shared launcher used by both public entry points
    in the ``physguard`` package.  The two entry points differ *only* in
    the ``mode`` argument they pass here:

        physguard.pretrain            -> launch(mode="pretrain")
        physguard.finetune_baselines  -> launch(mode="finetune")

    The mode is authoritative: ``args.is_finetune`` is set from it,
    ignoring any value that may have leaked in via the CLI.

    Args:
        mode: ``"pretrain"`` (Step 1, on simulated data) or ``"finetune"``
            (Step 2, on real data via DFT / L2-SP / EWC baselines).
    """
    if mode not in ("pretrain", "finetune"):
        raise ValueError(f"Unknown launch mode: {mode!r}")

    parser = argparse.ArgumentParser(description="Training Configurations")
    parser.add_argument("--config",          type=str,  default="configs/cylinder/fno.yaml")
    parser.add_argument("--train_data_type", type=str,  default="numerical",
                        help="numerical | real")
    parser.add_argument("--is_finetune",     action="store_true",
                        help="(auto-set by entry point; do not pass manually)")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint to resume training from (skips already-done iters)")
    parser.add_argument("--exp_path", type=str, default=None,
                        help="Override experiment output directory (use existing dir for resuming)")
    parser.add_argument("--use_hf_dataset",  action="store_true")
    parser.add_argument("--hf_auto_download",action="store_true")
    parser.add_argument("--hf_repo_id",      type=str,
                        default="AI4Science-WestlakeU/RealPDEBench")
    parser.add_argument("--hf_endpoint",     type=str,  default=None)
    parser.add_argument("--hf_revision",     type=str,  default=None)
    args = parser.parse_args()

    # The entry-point mode is authoritative.
    args.is_finetune = (mode == "finetune")

    # Resolve config path
    if not os.path.exists(args.config):
        candidate = os.path.join(os.path.dirname(__file__), args.config)
        if os.path.exists(candidate):
            args.config = candidate

    args = add_args_from_config(args, parser)

    # Set default value for ddp_port (can be overridden in yaml)
    if not hasattr(args, 'ddp_port'):
        args.ddp_port = 12355

    # ── Parse gpu field ──────────────────────────────────────────────────────
    gpu_ids    = parse_gpu_ids(args.gpu)
    world_size = len(gpu_ids)

    print(f"[Launch] mode={mode}  GPU config: {gpu_ids}  →  "
          f"{'DDP multi-GPU' if world_size > 1 else 'single GPU'} training")

    # ── Create experiment directory (before spawn, once only) ─────────────
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.is_finetune:
        exp_suffix = 'pretrained'
    else:
        reg_type = getattr(args, 'reg_type', 'none')
        if reg_type == 'l2':
            exp_suffix = 'l2sp'
        elif reg_type == 'ewc':
            exp_suffix = 'ewc'
        else:
            exp_suffix = 'dft'
    if getattr(args, 'exp_path', None):
        exp_path = args.exp_path
    else:
        exp_path = os.path.join(
            args.results_path, args.model_name,
            args.exp_name + '_' + exp_suffix,
            current_time
        )
    os.makedirs(exp_path, exist_ok=True)

    # ── Pre-compute normalizer stats (once, before spawning DDP workers) ────
    # In DDP mode the NCCL barrier has a fixed 600-second timeout, which is
    # far too short for computing mean/std over large datasets.  We therefore
    # run the normalizer on the main (CPU-only) process so the cache file is
    # written before any worker is spawned; workers will load from cache
    # instantly without any barrier or coordination.
    if world_size > 1 and args.normalizer != 'none':
        print("[Launch] Pre-computing normalizer stats on main process …")
        _, _, _norm_ds = build_datasets(args)
        _cpu = torch.device('cpu')
        if args.normalizer == 'gaussian':
            GaussianNormalizer(_norm_ds, device=_cpu)
        elif args.normalizer == 'range':
            RangeNormalizer(_norm_ds, device=_cpu)
        print("[Launch] Normalizer stats ready.")
        del _norm_ds

    # ── Launch training ───────────────────────────────────────────────────────
    if world_size == 1:
        # Single GPU: call directly, no spawn needed
        train(rank=0, gpu_ids=gpu_ids, args=args, exp_path=exp_path)
    else:
        # Multi-GPU: spawn world_size subprocesses
        mp.spawn(
            fn       = train,
            args     = (gpu_ids, args, exp_path),
            nprocs   = world_size,
            join     = True,
        )
