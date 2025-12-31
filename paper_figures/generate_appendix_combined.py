#!/usr/bin/env python3
"""
Generate NeurIPS appendix combined figures.

One figure per (dataset, model_architecture).

Layout (example for cylinder + FNO):
  Row 0   : Ground Truth       at t=3, t=53, t=103, t=153
  Row 1   : Pretrained          (same time steps)
  Row 2   : DFT
  Row 3   : EWC
  Row 4   : L2-SP
  Row 5   : PhysGuard

  - Column headers: time step labels
  - Row labels (left): method names
  - Each column shares a unified colorbar (same vmin/vmax across all rows)
  - Colorbar placed at the bottom of each column, height ~ 1 panel

Style: jet colormap + white streamlines (velocity magnitude |u|).

Usage:
  cd ./RealPDEBench
  conda run -n pytorch310 python ../scripts/generate_appendix_combined.py --gpu 2
"""

import sys, os, gc, argparse, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'RealPDEBench'))

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

# ─── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})

# ─── Project imports ─────────────────────────────────────────────────
from realpdebench.data.fluid_hf_dataset import CylinderHFDataset, ControlledCylinderHFDataset
from realpdebench.data.combustion_hf_dataset import CombustionHFDataset
from realpdebench.data.data_normalizer import GaussianNormalizer
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import _add_streamlines

# =====================================================================
#  CONFIGURATION  — Edit this section for different dataset/arch combos
# =====================================================================

_DATASET_ROOT = './data/realpdebench/'
_BASE_RESULTS_CYL = 'results/001-cylinder'
_BASE_RESULTS_CTRL = 'results/002-control_cylinder'
_BASE_RESULTS_COMB = 'results/003-combustion'

# --- Cylinder + FNO ---
CYLINDER_FNO = dict(
    dataset_name='cylinder',
    architecture='FNO',
    N_autoregressive=10,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='fno',
        modes1=4, modes2=12, modes3=16,
        n_layers=4, width=64,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CYL}/fno/fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth'),
        ('DFT',         f'{_BASE_RESULTS_CYL}/fno/fno_cylinder_dft/2026-03-10_03-40-24/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_CYL}/fno/fno_cylinder_ewcft/lam1.0/2026-03-12_20-09-12/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CYL}/fno/fno_cylinder_l2ft/lam0.0001/2026-03-14_11-45-57/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CYL}/fno/fno_cylinder_nsft/2026-04-06_02-27-58/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Cylinder + CNO ---
CYLINDER_CNO = dict(
    dataset_name='cylinder',
    architecture='CNO',
    N_autoregressive=3,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='cno',
        N_layers=3,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CYL}/cno/cno_cylinder_pretrained/2026-03-11_09-49-04/model_5000.pth'),
        ('DFT',         f'{_BASE_RESULTS_CYL}/cno/cno_cylinder_dft/2026-03-13_15-20-44/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_CYL}/cno/cno_cylinder_ewcft/2026-03-13_15-16-11/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CYL}/cno/cno_cylinder_l2ft/lam0.0001/2026-03-13_15-16-15/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CYL}/cno/cno_cylinder_nsft/2026-03-14_06-37-42/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Cylinder + DeepONet ---
CYLINDER_DEEPONET = dict(
    dataset_name='cylinder',
    architecture='DeepONet',
    N_autoregressive=10,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='deeponet',
        p=128,
        dropout_rate=0.1,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CYL}/deeponet/deeponet_cylinder_pretrained/2026-03-12_11-16-03/model_0100.pth'),
        ('DFT',         f'{_BASE_RESULTS_CYL}/deeponet/deeponet_cylinder_dft/2026-03-13_22-22-57/model_0560.pth'),
        ('EWC',         f'{_BASE_RESULTS_CYL}/deeponet/deeponet_cylinder_ewcft/lam1.0/2026-03-14_06-34-38/model_2720.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CYL}/deeponet/deeponet_cylinder_l2ft/lam0.0001/2026-03-14_06-34-38/model_2560.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CYL}/deeponet/deeponet_cylinder_nsft/2026-03-13_00-38-45/model_0080.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Cylinder + DPOT ---
CYLINDER_DPOT = dict(
    dataset_name='cylinder',
    architecture='DPOT',
    N_autoregressive=1,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='dpot',
        img_size=128,
        patch_size=8,
        in_channels=4,
        out_channels=4,
        in_timesteps=20,
        out_timesteps=20,
        embed_dim=1024,
        depth=6,
        n_blocks=8,
        modes=32,
        mlp_ratio=1,
        out_layer_dim=32,
        normalize=False,
        act='gelu',
        time_agg='exp_mlp',
        n_cls=12,
        model_type='dpot',
        checkpoint_path='./dpot_ckpts/model_S.pth',
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CYL}/dpot/dpot_s_cylinder_pretrained/2026-03-12_01-49-29/model_4000.pth'),
        ('DFT',         f'{_BASE_RESULTS_CYL}/dpot/dpot_s_cylinder_dft/2026-03-13_15-16-19/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_CYL}/dpot/dpot_s_cylinder_ewcft/lam1.0/2026-03-13_22-23-11/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CYL}/dpot/dpot_s_cylinder_l2ft/lam0.0001/2026-03-13_22-23-24/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CYL}/dpot/dpot_s_cylinder_nsft/2026-04-01_09-58-35/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Cylinder + Transolver ---
CYLINDER_TRANSOLVER = dict(
    dataset_name='cylinder',
    architecture='Transolver',
    N_autoregressive=3,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='transolver',
        space_dim=3,
        n_layers=1,
        n_hidden=256,
        n_head=8,
        H=128,
        W=64,
        D=20,
        fun_dim=0,
        out_dim=3,
        ref=4,
        dropout=0.1,
        act='gelu',
        mlp_ratio=4,
        slice_num=16,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CYL}/transolver/transolver_cylinder_pretrained/2026-03-25_21-23-52/model_5000.pth'),
        ('DFT',         f'{_BASE_RESULTS_CYL}/transolver/transolver_cylinder_dft/2026-03-26_10-11-26/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_CYL}/transolver/transolver_cylinder_ewcft/2026-03-28_15-40-56/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CYL}/transolver/transolver_cylinder_l2ft/2026-03-28_16-25-06/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CYL}/transolver/transolver_cylinder_nsft/2026-03-30_14-04-24/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# =====================================================================
#  CONTROLLED CYLINDER CONFIGURATIONS
# =====================================================================

# --- Controlled Cylinder + FNO ---
CTRL_CYL_FNO = dict(
    dataset_name='controlled_cylinder',
    architecture='FNO',
    N_autoregressive=10,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='fno',
        modes1=4, modes2=12, modes3=16,
        n_layers=4, width=64,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CTRL}/fno/fno_control_pretrained/2026-03-15_02-36-15/model_3840.pth'),
        ('DFT',         f'{_BASE_RESULTS_CTRL}/fno/fno_control_dft/2026-03-18_15-19-33/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_CTRL}/fno/fno_control_ewcft/2026-03-24_19-30-22/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CTRL}/fno/fno_control_l2ft/2026-03-26_00-48-45/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CTRL}/fno/fno_control_nsft/2026-04-05_22-56-53/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Controlled Cylinder + CNO ---
CTRL_CYL_CNO = dict(
    dataset_name='controlled_cylinder',
    architecture='CNO',
    N_autoregressive=3,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='cno',
        N_layers=3,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CTRL}/cno/cno_control_pretrained/2026-03-16_00-37-17/model_2600.pth'),
        ('DFT',         f'{_BASE_RESULTS_CTRL}/cno/cno_control_dft/2026-03-18_15-27-08/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_CTRL}/cno/cno_control_ewcft/2026-03-24_19-29-14/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CTRL}/cno/cno_control_l2ft/2026-03-26_00-53-56/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CTRL}/cno/cno_control_nsft/2026-03-18_20-47-52/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Controlled Cylinder + DeepONet ---
CTRL_CYL_DEEPONET = dict(
    dataset_name='controlled_cylinder',
    architecture='DeepONet',
    N_autoregressive=10,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='deeponet',
        p=256,
        dropout_rate=0.1,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CTRL}/deeponet/deeponet_control_pretrained/2026-03-16_11-38-33/model_1000.pth'),
        ('DFT',         f'{_BASE_RESULTS_CTRL}/deeponet/deeponet_control_dft/2026-03-18_15-27-37/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_CTRL}/deeponet/deeponet_control_ewcft/2026-03-25_11-38-46/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CTRL}/deeponet/deeponet_control_l2ft/2026-03-26_11-10-31/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CTRL}/deeponet/deeponet_control_nsft/2026-03-18_20-48-35/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Controlled Cylinder + DPOT ---
CTRL_CYL_DPOT = dict(
    dataset_name='controlled_cylinder',
    architecture='DPOT',
    N_autoregressive=1,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='dpot',
        img_size=128,
        patch_size=8,
        in_channels=5,
        out_channels=4,
        in_timesteps=10,
        out_timesteps=10,
        embed_dim=1024,
        depth=6,
        n_blocks=8,
        modes=32,
        mlp_ratio=1,
        out_layer_dim=32,
        normalize=False,
        act='gelu',
        time_agg='exp_mlp',
        n_cls=12,
        model_type='dpot',
        checkpoint_path='./dpot_ckpts/model_S.pth',
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CTRL}/dpot/dpot_s_control_pretrained/2026-03-17_19-52-05/model_0320.pth'),
        ('DFT',         f'{_BASE_RESULTS_CTRL}/dpot/dpot_s_control_dft/2026-03-18_15-28-04/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_CTRL}/dpot/dpot_s_control_ewcft/2026-03-25_11-47-38/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CTRL}/dpot/dpot_s_control_l2ft/2026-03-26_09-19-34/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CTRL}/dpot/dpot_s_control_nsft/2026-03-31_03-48-52/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Controlled Cylinder + Transolver ---
CTRL_CYL_TRANSOLVER = dict(
    dataset_name='controlled_cylinder',
    architecture='Transolver',
    N_autoregressive=2,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='transolver',
        space_dim=5,
        n_layers=1,
        n_hidden=256,
        n_head=8,
        H=64,
        W=128,
        D=10,
        fun_dim=0,
        out_dim=3,
        ref=4,
        dropout=0.1,
        act='gelu',
        mlp_ratio=4,
        slice_num=16,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_CTRL}/transolver/transolver_control_pretrained/2026-03-26_19-32-01/model_5000.pth'),
        ('DFT',         f'{_BASE_RESULTS_CTRL}/transolver/transolver_control_dft/2026-03-28_03-29-58/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_CTRL}/transolver/transolver_control_ewcft/2026-03-28_03-31-03/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_CTRL}/transolver/transolver_control_l2ft/2026-03-28_16-25-16/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_CTRL}/transolver/transolver_control_nsft/2026-03-30_15-21-17/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# =====================================================================
#  COMBUSTION CONFIGURATIONS
# =====================================================================

# --- Combustion + FNO ---
COMBUSTION_FNO = dict(
    dataset_name='combustion',
    architecture='FNO',
    N_autoregressive=1,
    mask_prob=0.5,
    model_kwargs=dict(
        model_name='fno',
        modes1=4, modes2=16, modes3=16,
        n_layers=4, width=64,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_COMB}/fno/fno_combustion_pretrained/2026-03-20_11-54-28/model_1000.pth'),
        ('DFT',         f'{_BASE_RESULTS_COMB}/fno/fno_combustion_dft/2026-03-22_12-19-34/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_COMB}/fno/fno_combustion_ewcft/2026-03-24_21-54-32/model_3920.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_COMB}/fno/fno_combustion_l2ft/2026-03-26_10-29-14/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_COMB}/fno/fno_combustion_nsft/2026-03-22_13-13-42/model_3840.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Combustion + CNO ---
COMBUSTION_CNO = dict(
    dataset_name='combustion',
    architecture='CNO',
    N_autoregressive=3,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='cno',
        N_layers=3,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_COMB}/cno/cno_combustion_pretrained/2026-03-20_21-35-00/model_5000.pth'),
        ('DFT',         f'{_BASE_RESULTS_COMB}/cno/cno_combustion_dft/2026-03-22_12-19-54/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_COMB}/cno/cno_combustion_ewcft/2026-03-25_19-48-42/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_COMB}/cno/cno_combustion_l2ft/2026-03-26_15-19-28/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_COMB}/cno/cno_combustion_nsft/2026-04-01_14-07-51/model_6000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Combustion + DeepONet ---
COMBUSTION_DEEPONET = dict(
    dataset_name='combustion',
    architecture='DeepONet',
    N_autoregressive=1,
    mask_prob=0.5,
    model_kwargs=dict(
        model_name='deeponet',
        p=128,
        dropout_rate=0.1,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_COMB}/deeponet/deeponet_combustion_pretrained/2026-03-21_10-20-54/model_3000.pth'),
        ('DFT',         f'{_BASE_RESULTS_COMB}/deeponet/deeponet_combustion_dft/2026-03-22_12-20-30/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_COMB}/deeponet/deeponet_combustion_ewcft/2026-03-25_20-02-01/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_COMB}/deeponet/deeponet_combustion_l2ft/2026-03-26_15-21-47/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_COMB}/deeponet/deeponet_combustion_nsft/2026-03-27_20-56-15/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Combustion + DPOT ---
COMBUSTION_DPOT = dict(
    dataset_name='combustion',
    architecture='DPOT',
    N_autoregressive=1,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='dpot',
        img_size=128,
        patch_size=8,
        in_channels=16,
        out_channels=16,
        in_timesteps=20,
        out_timesteps=20,
        embed_dim=1024,
        depth=6,
        n_blocks=8,
        modes=32,
        mlp_ratio=1,
        out_layer_dim=32,
        normalize=False,
        act='gelu',
        time_agg='exp_mlp',
        n_cls=12,
        model_type='dpot',
        checkpoint_path='./dpot_ckpts/model_S.pth',
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_COMB}/dpot/dpot_s_combustion_pretrained/2026-03-21_17-01-45/model_10000.pth'),
        ('DFT',         f'{_BASE_RESULTS_COMB}/dpot/dpot_s_combustion_dft/2026-03-22_12-20-48/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_COMB}/dpot/dpot_s_combustion_ewcft/2026-03-25_20-02-16/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_COMB}/dpot/dpot_s_combustion_l2ft/2026-03-26_16-05-18/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_COMB}/dpot/dpot_s_combustion_nsft/2026-03-24_01-54-14/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Combustion + Transolver ---
COMBUSTION_TRANSOLVER = dict(
    dataset_name='combustion',
    architecture='Transolver',
    N_autoregressive=1,
    mask_prob=0.1,
    model_kwargs=dict(
        model_name='transolver',
        space_dim=16,
        n_layers=1,
        n_hidden=256,
        n_head=8,
        H=64,
        W=64,
        D=20,
        fun_dim=0,
        out_dim=16,
        ref=4,
        dropout=0.1,
        act='gelu',
        mlp_ratio=4,
        slice_num=16,
    ),
    methods=[
        ('Pretrained',  f'{_BASE_RESULTS_COMB}/transolver/transolver_combustion_pretrained/2026-03-26_19-32-07/model_5000.pth'),
        ('DFT',         f'{_BASE_RESULTS_COMB}/transolver/transolver_combustion_dft/2026-03-28_03-31-03/model_4000.pth'),
        ('EWC',         f'{_BASE_RESULTS_COMB}/transolver/transolver_combustion_ewcft/2026-03-28_03-31-03/model_4000.pth'),
        ('L2-SP',       f'{_BASE_RESULTS_COMB}/transolver/transolver_combustion_l2ft/2026-03-28_16-25-25/model_4000.pth'),
        ('PhysGuard',   f'{_BASE_RESULTS_COMB}/transolver/transolver_combustion_nsft/2026-03-31_00-57-28/model_4000.pth'),
    ],
    sample_idx=0,
    dataset_root=_DATASET_ROOT,
)

# --- Registry ---
ALL_CONFIGS = {
    'cylinder_fno': CYLINDER_FNO,
    'cylinder_cno': CYLINDER_CNO,
    'cylinder_deeponet': CYLINDER_DEEPONET,
    'cylinder_dpot': CYLINDER_DPOT,
    'cylinder_transolver': CYLINDER_TRANSOLVER,
    'ctrl_cyl_fno': CTRL_CYL_FNO,
    'ctrl_cyl_cno': CTRL_CYL_CNO,
    'ctrl_cyl_deeponet': CTRL_CYL_DEEPONET,
    'ctrl_cyl_dpot': CTRL_CYL_DPOT,
    'ctrl_cyl_transolver': CTRL_CYL_TRANSOLVER,
    'comb_fno': COMBUSTION_FNO,
    'comb_cno': COMBUSTION_CNO,
    'comb_deeponet': COMBUSTION_DEEPONET,
    'comb_dpot': COMBUSTION_DPOT,
    'comb_transolver': COMBUSTION_TRANSOLVER,
}

# =====================================================================
#  Helpers
# =====================================================================

def build_datasets(cfg, N_autoregressive):
    """Build test dataset and normalizer dataset."""
    ds_name = cfg['dataset_name']
    if ds_name == 'controlled_cylinder':
        DatasetClass = ControlledCylinderHFDataset
    elif ds_name == 'combustion':
        DatasetClass = CombustionHFDataset
    else:
        DatasetClass = CylinderHFDataset

    test_ds = DatasetClass(
        dataset_name=ds_name,
        dataset_root=cfg['dataset_root'],
        dataset_type='real',
        mode='test',
        N_autoregressive=N_autoregressive,
        mask_prob=cfg.get('mask_prob', 0.1),
    )
    # Normalizer always uses N_autoregressive=1 (single-step), numerical data
    norm_ds = DatasetClass(
        dataset_name=ds_name,
        dataset_root=cfg['dataset_root'],
        dataset_type='numerical',
        mode='train',
        N_autoregressive=1,
        mask_prob=cfg.get('mask_prob', 0.1),
    )
    return test_ds, norm_ds


def build_model(cfg, norm_ds, device):
    """Build the model from config kwargs."""
    kwargs = dict(cfg['model_kwargs'])
    # For most models, checkpoint_path=None. DPOT needs its pretrained backbone path.
    if 'checkpoint_path' not in kwargs:
        kwargs['checkpoint_path'] = None
    return load_model(norm_ds, device=device, **kwargs)


def run_inference(model, normalizer, input_tensor, N_autoregressive, device, target_c):
    """Autoregressive inference → (1, N_AR*out_step, H, W, C) in original scale.
    
    Handles parameter-conditioned datasets (e.g. controlled_cylinder) where
    input channels > target channels.
    """
    model.eval()
    with torch.no_grad():
        t_in = input_tensor.shape[1]
        
        # Detect parameter conditioning: input has more channels than output
        in_c = input_tensor.shape[-1]
        in_control = False
        para_c = 0
        if in_c > target_c:
            para_c = in_c - target_c
            para_input = input_tensor[..., -para_c:]  # (1, T_in, H, W, para_c)
            in_control = True

        dummy = torch.zeros(1, t_in, *input_tensor.shape[2:-1], target_c).to(device)
        inp, _ = normalizer.preprocess(input_tensor.to(device), dummy)

        preds = [inp]
        for _ in range(N_autoregressive):
            p = model(preds[-1])
            _, p = normalizer.postprocess(preds[-1], p)
            if in_control:
                p = torch.cat([p, para_input.to(p.device)], dim=-1)
            p, _ = normalizer.preprocess(p, dummy)
            preds.append(p)

        pred = torch.cat(preds[1:], dim=1)
        if in_control:
            pred = pred[..., :-para_c]
        _, pred = normalizer.postprocess(input_tensor[..., :target_c].to(device), pred)
    return pred.cpu()


def compute_time_indices(total_t, n_cols=4):
    """Same formula as eval.py's plot_result: spread n_cols across total_t."""
    return [total_t // n_cols * k + (total_t - 1) % n_cols for k in range(n_cols)]


def make_figure(gt_np, preds_dict, method_names, t_indices, output_path,
                dataset_label, arch_label, is_combustion=False):
    """
    Create the combined appendix figure.

    Args:
        gt_np:         (T, H, W, C) ground truth numpy array
        preds_dict:    {method_name: (T, H, W, C) numpy}
        method_names:  list of method names in order
        t_indices:     list of time step indices (columns)
        output_path:   path to save PDF/PNG
        dataset_label: e.g. "Cylinder Flow"
        arch_label:    e.g. "FNO"
        is_combustion: if True, plot scalar channel 0 with viridis, no streamlines
    """
    n_methods = len(method_names)
    n_rows = 1 + n_methods  # GT + methods
    n_cols = len(t_indices)
    row_labels = ['Ground Truth'] + method_names

    h, w = gt_np.shape[1], gt_np.shape[2]
    panel_w = 1.45   # inches per panel
    panel_h = panel_w * h / w  # maintain aspect ratio
    cbar_h = 0.12    # colorbar strip height

    fig_w = panel_w * n_cols + 1.0   # extra space for row labels + right margin
    fig_h = panel_h * n_rows + cbar_h * n_cols + 0.8  # panels + colorbars + title

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Use gridspec: n_rows+1 rows (data rows + colorbar row), n_cols columns
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        figure=fig,
        height_ratios=[1.0] * n_rows + [0.04],
        hspace=0.08,
        wspace=0.05,
        left=0.09, right=0.97,
        top=0.92, bottom=0.03,
    )

    cmap = 'jet'

    # ── 1. Compute unified vmin/vmax per column ──
    col_vmins = []
    col_vmaxs = []
    for col, t in enumerate(t_indices):
        if is_combustion:
            gt_field = gt_np[t, :, :, 0]
        else:
            gt_field = np.sqrt(gt_np[t, :, :, 0]**2 + gt_np[t, :, :, 1]**2)
        vmin = float(gt_field.min())
        vmax = float(gt_field.max())
        for name in method_names:
            pred = preds_dict[name]
            if is_combustion:
                f = pred[t, :, :, 0]
            else:
                f = np.sqrt(pred[t, :, :, 0]**2 + pred[t, :, :, 1]**2)
            vmin = min(vmin, float(f.min()))
            vmax = max(vmax, float(f.max()))
        col_vmins.append(vmin)
        col_vmaxs.append(vmax)

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    X, Y = np.meshgrid(x, y)

    # ── 2. Plot all panels ──
    for row in range(n_rows):
        for col, t in enumerate(t_indices):
            ax = fig.add_subplot(gs[row, col])

            # Get field data
            if row == 0:
                data = gt_np[t]
            else:
                data = preds_dict[method_names[row - 1]][t]

            if is_combustion:
                field = data[:, :, 0]
            else:
                ux = data[:, :, 0]
                uy = data[:, :, 1]
                field = np.sqrt(ux**2 + uy**2)

            vmin = col_vmins[col]
            vmax = col_vmaxs[col]

            # Contourf
            cf = ax.contourf(X, Y, field, levels=64,
                             cmap=cmap, vmin=vmin, vmax=vmax)
            # Streamlines (only for flow datasets)
            if not is_combustion:
                _add_streamlines(ax, ux, uy, density=0.8,
                                 color='white', linewidth=0.35)

            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column header (time labels on top row)
            if row == 0:
                ax.set_title(f'$t = {t}$', fontsize=9, fontweight='bold', pad=4)

            # Row label (method name on leftmost column)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8, fontweight='bold',
                              labelpad=3, rotation=90)

    # ── 3. Colorbars (one per column, at the bottom) ──
    for col, t in enumerate(t_indices):
        cax = fig.add_subplot(gs[n_rows, col])
        vmin = col_vmins[col]
        vmax = col_vmaxs[col]
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=6)
        # Only show 4-5 ticks
        cb.locator = plt.MaxNLocator(nbins=4)
        cb.update_ticks()

    # ── 4. Suptitle ──
    fig.suptitle(f'{dataset_label} — {arch_label}',
                 fontsize=11, fontweight='bold', y=0.98)

    # ── 5. Save ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    logging.info(f"Saved: {output_path}")
    logging.info(f"Saved: {png_path}")


# =====================================================================
#  Main
# =====================================================================

def run_single_config(cfg, device, sample_idx, n_ar_override=None):
    """Run a single (dataset, architecture) config and produce the appendix figure."""
    N_AR = n_ar_override if n_ar_override is not None else cfg['N_autoregressive']

    logging.info(f"\n{'='*60}")
    logging.info(f"  {cfg['dataset_name']} — {cfg['architecture']}  (N_AR={N_AR})")
    logging.info(f"{'='*60}")

    # ── Build datasets ──
    logging.info("Loading datasets...")
    test_ds, norm_ds = build_datasets(cfg, N_AR)
    normalizer = GaussianNormalizer(norm_ds, device=device)

    # ── Build model ──
    logging.info("Building model...")
    model = build_model(cfg, norm_ds, device)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params:,}")

    # ── Get test sample ──
    inp, tgt = test_ds[sample_idx]
    inp_batch = inp.unsqueeze(0)  # (1, T_in, H, W, C)
    gt_np = tgt.numpy()           # (T_out, H, W, C)
    target_c = tgt.shape[-1]      # number of physics channels
    total_t = gt_np.shape[0]
    t_indices = compute_time_indices(total_t)
    logging.info(f"Sample {sample_idx}: GT shape={gt_np.shape}, "
                 f"input shape={inp.shape}, t_indices={t_indices}")

    # ── Run inference for each method ──
    preds_dict = {}
    method_names = []
    for name, ckpt_path in cfg['methods']:
        logging.info(f"[{name}] Loading checkpoint: {ckpt_path}")
        model.load_checkpoint(ckpt_path, device)
        pred = run_inference(model, normalizer, inp_batch,
                             N_AR, device, target_c)
        preds_dict[name] = pred[0].numpy()  # (T_out, H, W, C)
        method_names.append(name)
        logging.info(f"[{name}] Prediction shape: {pred.shape}")

    # ── Free GPU memory ──
    del model, normalizer, test_ds, norm_ds
    torch.cuda.empty_cache()
    gc.collect()

    # ── Generate figure ──
    dataset_label = {
        'cylinder': 'Cylinder Flow',
        'controlled_cylinder': 'Controlled Cylinder',
        'combustion': 'Combustion',
    }.get(cfg['dataset_name'], cfg['dataset_name'])
    arch_label = cfg['architecture']

    output_dir = './figures/appendix'
    ar_suffix = f'_1step' if (n_ar_override is not None and n_ar_override == 1) else ''
    output_path = os.path.join(
        output_dir,
        f"appendix_{cfg['dataset_name']}_{cfg['architecture'].lower()}_sample{sample_idx:03d}{ar_suffix}.pdf"
    )

    if n_ar_override is not None and n_ar_override == 1:
        arch_label = f"{arch_label} (1-step)"

    make_figure(
        gt_np=gt_np,
        preds_dict=preds_dict,
        method_names=method_names,
        t_indices=t_indices,
        output_path=output_path,
        dataset_label=dataset_label,
        arch_label=arch_label,
        is_combustion=(cfg['dataset_name'] == 'combustion'),
    )


def main():
    all_config_names = list(ALL_CONFIGS.keys())
    valid_choices = all_config_names + ['all_cylinder', 'all_ctrl_cyl', 'all_comb']

    parser = argparse.ArgumentParser(description="Generate appendix combined figures")
    parser.add_argument('--gpu', type=int, default=2,
                        help='GPU device index (default: 2)')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Test sample index to visualize')
    parser.add_argument('--config', type=str, default='cylinder_fno',
                        choices=valid_choices,
                        help='Which configuration to run, or "all_cylinder" / "all_ctrl_cyl" for batch')
    parser.add_argument('--n_ar', type=int, default=None,
                        help='Override N_autoregressive (e.g. 1 for single-step short-horizon)')
    cmd_args = parser.parse_args()

    device = torch.device(f'cuda:{cmd_args.gpu}' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Determine which configs to run
    if cmd_args.config == 'all_cylinder':
        configs_to_run = [k for k in all_config_names if k.startswith('cylinder_')]
    elif cmd_args.config == 'all_ctrl_cyl':
        configs_to_run = [k for k in all_config_names if k.startswith('ctrl_cyl_')]
    elif cmd_args.config == 'all_comb':
        configs_to_run = [k for k in all_config_names if k.startswith('comb_')]
    else:
        configs_to_run = [cmd_args.config]

    for config_name in configs_to_run:
        cfg = ALL_CONFIGS[config_name]
        run_single_config(cfg, device, cmd_args.sample_idx, n_ar_override=cmd_args.n_ar)

    logging.info("\n*** All done! ***")


if __name__ == '__main__':
    main()
