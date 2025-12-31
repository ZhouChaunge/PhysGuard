# PhysGuard: Fisher-Guided Gradient Projection for Sim-to-Real Neural PDE Surrogates

This repository contains the official implementation of **PhysGuard**, a
physics-preserving framework for sim-to-real adaptation of neural operator
PDE surrogates. PhysGuard identifies a low-dimensional, physics-critical
parameter subspace via the empirical Fisher Information Matrix (FIM) computed
on simulation data, and constrains fine-tuning gradients to its orthogonal
complement so that the low-frequency PDE structures learned during
pre-training are not overwritten when adapting to real experimental data.

The implementation is built on top of
[RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench), the
benchmark used in our experiments. The PhysGuard contribution is contained in
`RealPDEBench/realpdebench/nullspace/` and `RealPDEBench/realpdebench/train_nullspace.py`;
all other files in `RealPDEBench/` are derived from the upstream benchmark
(see `RealPDEBench/LICENSE`).

> Manuscript: see `../001-manuscript/neurips_2026.tex` (companion to this code).

---

## Repository layout

```
002-code/
├── README.md                       this file
├── .gitignore
├── RealPDEBench/                   benchmark + PhysGuard package
│   ├── pyproject.toml
│   ├── environment.yml
│   ├── LICENSE                     CC BY-NC 4.0 (RealPDEBench)
│   ├── README.md                   upstream benchmark documentation
│   └── realpdebench/
│       ├── nullspace/              <-- PhysGuard core (Fisher-guided projection)
│       │   ├── null_space_projector.py
│       │   └── null_space_optimizer.py
│       ├── train_nullspace.py      <-- PhysGuard fine-tuning entry point
│       ├── train_surrogate.py      simulated/real training (baseline)
│       ├── train_gpus.py           multi-GPU training driver
│       ├── eval.py                 evaluation entry point
│       ├── configs/                YAML configs (per scenario × architecture × method)
│       │   ├── cylinder/   *_nullspace.yaml, *_finetune_{real,ewc,l2}.yaml, ...
│       │   ├── controlled_cylinder/
│       │   ├── combustion/
│       │   ├── fsi/  ·  foil/
│       │   └── ablation/cylinder_deeponet/{E4_alpha,E5_tau}/
│       ├── model/                  FNO, CNO, DeepONet, Transolver, DPOT, ...
│       ├── data/                   dataset loaders (HF Arrow + HDF5)
│       └── utils/                  metrics, normalisers, helpers
├── analysis/                       Section 4.3 / 4.4 analyses (training curves,
│   ├── sec43_training_curves.py    Fisher subspace, eigenspectrum, gradient
│   ├── sec44_eigenspectrum.py      overlap, subspace dimension)
│   ├── sec44_fisher_subspace.py
│   ├── sec44_gradient_overlap.py
│   └── sec44_subspace_dim.py
└── paper_figures/                  scripts to reproduce manuscript figures
    ├── generate_figure1.py         Fig. 1  (motivation / overview)
    ├── generate_motivation_fig*.py Fig. 1 variants
    ├── rq1_*.py                    Sec. 4.4 RQ1 figures (Fisher–frequency alignment,
    │                               cross-architecture FIM, intuitive viz, ...)
    ├── generate_freq_analysis*.py  spectral comparisons
    ├── generate_qualitative_*.py   qualitative prediction plots
    ├── generate_appendix_*.py      appendix / universality figures
    ├── plot_E4_alpha_ablation.py   ablation: protection strength α
    └── eval_*.sh                   batch-evaluation shell drivers
```

The trained checkpoints, raw experimental result tables, dataset shards, and
high-resolution figure outputs are **not** included in this repository to keep
it small. Datasets and pre-trained backbones can be obtained from the public
sources listed below.

---

## Installation

Python ≥ 3.10 is required.

```bash
git clone <this-repo>
cd 002-code/RealPDEBench
pip install -e .
```

This installs the `realpdebench` package together with the PhysGuard
extensions. A reference conda environment is provided in
`RealPDEBench/environment.yml`.

---

## Data and pre-trained checkpoints

PhysGuard is benchmarked on
[RealPDEBench](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench),
which provides paired numerical and real-world trajectories for `cylinder`,
`controlled_cylinder`, `fsi`, `foil`, and `combustion`.

```bash
# Metadata only (safe default)
realpdebench download --dataset-root ./data/realpdebench --scenario cylinder --what metadata

# Full HF Arrow shards (large)
realpdebench download --dataset-root ./data/realpdebench --scenario cylinder \
    --what hf_dataset --dataset-type real     --endpoint https://hf-mirror.com
realpdebench download --dataset-root ./data/realpdebench --scenario cylinder \
    --what hf_dataset --dataset-type numerical --endpoint https://hf-mirror.com
```

Pre-trained backbones for all (architecture × scenario × paradigm)
combinations are released under
[`AI4Science-WestlakeU/RealPDEBench-models`](https://huggingface.co/AI4Science-WestlakeU/RealPDEBench-models).
DPOT additionally requires its own pretrained weights:

```bash
python -m realpdebench.utils.dpot_ckpts_dl
```

After downloading data and checkpoints, edit the relevant YAML config to
point `dataset_root` and `checkpoint_path` to your local directories.

---

## Reproducing the main experiments

All commands assume the working directory is `002-code/RealPDEBench/`.

### 1. Pre-training on simulation data (skip if using released backbones)

```bash
python -m realpdebench.train_surrogate \
    --config configs/cylinder/fno.yaml \
    --train_data_type numerical
```

### 2. PhysGuard sim-to-real fine-tuning

The PhysGuard entry point is `train_nullspace.py`. It executes the two-phase
algorithm of the manuscript: (i) compute the Fisher subspace from the
simulation set, (ii) fine-tune on real data with gradients projected onto the
orthogonal complement.

```bash
python -m realpdebench.train_nullspace \
    --config configs/cylinder/fno_nullspace.yaml
```

Key PhysGuard hyperparameters (also exposed as CLI flags):

| Flag                       | Meaning                                                                 | Default |
|----------------------------|-------------------------------------------------------------------------|---------|
| `--ns_variance_threshold`  | Adaptive `k` per layer: keep eigenvectors covering this Fisher fraction | `0.9`   |
| `--ns_n_components`        | Hard cap on `k` when adaptive selection is active                       | `200`   |
| `--ns_alpha`               | Projection strength α (1.0 = full null-space projection)                | `1.0`   |
| `--ns_max_samples`         | # simulation samples used to estimate the empirical FIM                 | `200`   |
| `--ns_protected_layers`    | Comma-separated substrings of parameter names to protect (default: all) | `None`  |
| `--ns_progressive`         | Linearly relax α towards `--ns_beta_min` over training                  | `False` |

### 3. Baseline fine-tuning protocols

For fair comparisons, the same hyperparameters are reused across protocols:

```bash
# Direct fine-tuning on real data
python -m realpdebench.train_surrogate \
    --config configs/cylinder/fno_finetune_real.yaml --train_data_type real --is_finetune

# Elastic Weight Consolidation (EWC)
python -m realpdebench.train_surrogate \
    --config configs/cylinder/fno_finetune_ewc.yaml  --train_data_type real --is_finetune

# L2-SP regularisation
python -m realpdebench.train_surrogate \
    --config configs/cylinder/fno_finetune_l2.yaml   --train_data_type real --is_finetune
```

The same pattern applies to all four scenarios in the paper
(`cylinder`, `controlled_cylinder`, `combustion`, plus the appendix scenarios)
and all four architectures (`fno`, `cno`, `deeponet`, `transolver`).
Configs follow the naming convention
`<arch>_{nullspace|finetune_real|finetune_ewc|finetune_l2}.yaml`.

### 4. Evaluation

```bash
python -m realpdebench.eval \
    --config configs/cylinder/fno_nullspace.yaml \
    --checkpoint_path /path/to/adapted_model.pth
```

Batch evaluation drivers used to populate the result tables in the paper are
in `paper_figures/eval_*.sh`.

### 5. Ablations (manuscript Section 4 / Appendix)

Configs for the protection-strength sweep (E4: α ∈ {0.3, 0.5, 0.7, 1.0}) and
the variance-threshold sweep (E5: τ ∈ {0.80, 0.85, 0.95, 0.99}) are in
`configs/ablation/cylinder_deeponet/`. After training, plot with:

```bash
python paper_figures/plot_E4_alpha_ablation.py
```

---

## Reproducing the figures

The `paper_figures/` directory contains the scripts used to generate every
figure in the manuscript and appendix. Most scripts cache their numerical
inputs to `.npz`/`.pt` files (not bundled), so you will need either the
trained checkpoints from step 2/3 above or the cached intermediate arrays
listed in the manuscript's reproducibility appendix.

Indicative entry points:

| Script                                  | Manuscript reference                       |
|-----------------------------------------|--------------------------------------------|
| `generate_figure1.py`, `generate_motivation_fig_v5.py` | Fig. 1 (motivation)        |
| `generate_freq_analysis_v3.py`          | Fig. 4 (spectral analysis)                 |
| `generate_qualitative_*.py`             | Fig. 3 (qualitative comparison)            |
| `rq1_fim_frequency_alignment.py`        | Sec. 4.4 — Fisher / frequency alignment    |
| `rq1_cross_arch_fim.py`                 | Sec. 4.4 — cross-architecture FIM evidence |
| `rq1_neurips_proof.py`, `rq1_direct_proof.py`         | Sec. 4.4 — empirical proofs   |
| `analysis/sec44_eigenspectrum.py`       | Sec. 4.4 — Fisher eigenspectrum            |
| `analysis/sec44_subspace_dim.py`        | Sec. 4.4 — adaptive subspace dimension     |

---

## Citing this work

If you use PhysGuard, please cite the accompanying manuscript (see
`../001-manuscript/`) and the underlying RealPDEBench benchmark:

```bibtex
@inproceedings{hu2026realpdebench,
  title={RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data},
  author={Hu, Peiyan and Feng, Haodong and Liu, Hongyuan and others},
  booktitle={ICLR},
  year={2026}
}
```

---

## License

- The PhysGuard additions (the `nullspace/` package, `train_nullspace.py`,
  the `analysis/` scripts, and the `paper_figures/` scripts) are released
  under the MIT license — see `LICENSE`.
- The underlying RealPDEBench code in `RealPDEBench/` is governed by
  `RealPDEBench/LICENSE` (CC BY-NC 4.0). All redistributions of that code
  must comply with its terms.
