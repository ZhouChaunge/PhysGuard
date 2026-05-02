# PhysGuard

Code for **PhysGuard: Fisher-Guided Gradient Projection for Sim-to-Real Neural PDE Surrogates**.

PhysGuard preserves the physics learned during simulation pre-training when adapting
a neural PDE surrogate to real experimental data. It uses the empirical Fisher
Information Matrix (FIM) of the pre-trained model to identify the parameter
directions that encode low-frequency physical structure, and then constrains
fine-tuning gradients to the **null space** of those directions. The whole
procedure adds no penalty term, no extra trainable module, and only one
hyperparameter — the protection strength `α`.

<p align="center">
  <img src="figures/method.png" alt="PhysGuard method overview" width="850"/>
</p>

## 📖 Method at a glance

Given a pre-trained neural operator `f_θ*` and a small batch of `N` simulation
gradients `g_i = ∇_θ ℓ(θ*; x_i, y_i)` stacked into `G ∈ R^{N×d}`:

1. **Subspace estimation (offline, once).** For every layer `m`, build the
   Gram matrix `K = G Gᵀ ∈ R^{N×N}`, run an SVD, keep the top `k_m`
   eigenvectors that capture `τ` (default `0.9`) of the cumulative Fisher
   variance, and lift them back to parameter space:

   ```
   U^(m) = normalise( Gᵀ V_{k_m} ) ∈ R^{d×k_m}
   ```

2. **Constrained fine-tuning (every step).** During real-data fine-tuning,
   project each per-layer gradient onto the safe subspace before the
   optimiser step:

   ```
   g_proj = g − α · U Uᵀ g          (α = 1.0 ⇒ full null-space projection)
   ```

The two functions are implemented in [`physguard/projector.py`](physguard/projector.py)
(subspace estimation) and [`physguard/optimizer.py`](physguard/optimizer.py)
(per-step gradient projection wrapper around any `torch.optim.Optimizer`).

## 🗂 Repository layout

```
.
├── physguard/                       core algorithm + experiment entry points
│   ├── projector.py                 NullSpaceProjector — Gram-trick FIM SVD
│   ├── optimizer.py                 NullSpaceOptimizer — gradient-projection wrapper
│   ├── _engine.py                   shared training engine (used by the entries below)
│   ├── pretrain.py                  Step 1 — pre-train on simulation data
│   ├── finetune_baselines.py        Step 2 — DFT / L2-SP / EWC baselines
│   └── finetune_physguard.py        Step 2 — PhysGuard (ours ★)
├── benchmark/                       benchmark engine (data loaders, models, eval)
│   ├── eval.py                      evaluation entry point
│   ├── model/                       FNO, CNO, DeepONet, Transolver
│   ├── data/                        HF Arrow dataset wrappers
│   └── utils/                       metrics, normalisers, helpers
├── configs/                         YAML configs — 3 scenarios × 4 archs × 5 paradigms
│   ├── 1-cylinder/
│   │   ├── fno/
│   │   │   ├── 1_pretrain.yaml      ← Step 1: pre-train on simulation
│   │   │   ├── 2_dft.yaml           ┐
│   │   │   ├── 2_ewc.yaml           │ Step 2: pick one fine-tuning method
│   │   │   ├── 2_l2sp.yaml          │
│   │   │   └── 2_physguard.yaml     ┘
│   │   ├── cno/
│   │   ├── deeponet/
│   │   └── transolver/
│   ├── 2-controlled_cylinder/       (same layout)
│   └── 3-combustion/                (same layout)
├── figures/                         method overview & motivation figures
├── pyproject.toml                   pip-installable package
├── environment.yml                  conda environment (non-torch deps)
├── install.sh                       auto-detects CUDA and installs PyTorch (Linux/macOS)
└── install.ps1                      same, for Windows (PowerShell)
```

---

## 🛠️ Step 0 — Environment Setup

> **Requirements:** Python ≥ 3.10, CUDA ≥ 11.8, one GPU (RTX 4090 / A40 / A100 is sufficient).

### Step 0.1 — Clone the repository

```bash
git clone https://github.com/<anonymous>/PhysGuard.git
cd PhysGuard
```

### Step 0.2 — Create the conda environment

This installs all dependencies **except** PyTorch (handled in the next step):

```bash
conda env create -f environment.yml
conda activate physguard
```

### Step 0.3 — Install PyTorch (auto-detects CUDA version)

**Linux / macOS:**
```bash
bash install.sh
```

**Windows (PowerShell):**
```powershell
.\install.ps1
```

Both scripts query `nvidia-smi` to detect the driver CUDA version and
install the matching `torch` wheel (cu118 / cu121 / cu124 / CPU-only).
They also run `pip install -e .` to register the `physguard` and `benchmark`
packages in editable mode.

### Step 0.4 — Verify the installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import physguard; import benchmark; print('OK')"
```

---

## ⏬ Step 1 — Download the Dataset

The experiments use [RealPDEBench](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench),
which provides paired numerical (simulation) and real-world trajectories for three scenarios.

The `benchmark download` CLI (installed by `pip install -e .`) handles
authentication, chunked transfer, and Arrow shard assembly automatically.

> **Tip:** Set `--endpoint https://hf-mirror.com` if you are behind the GFW.  
> Set `HF_HUB_DISABLE_XET=1` to avoid potential XET transport issues.

### Scenario 1 — Cylinder flow

```bash
# numerical (simulation) data — used for pre-training
benchmark download --dataset-root ./dataset --scenario cylinder \
    --what hf_dataset --dataset-type numerical

# real-world data — used for fine-tuning & evaluation
benchmark download --dataset-root ./dataset --scenario cylinder \
    --what hf_dataset --dataset-type real
```

### Scenario 2 — Controlled cylinder flow

```bash
benchmark download --dataset-root ./dataset --scenario controlled_cylinder \
    --what hf_dataset --dataset-type numerical

benchmark download --dataset-root ./dataset --scenario controlled_cylinder \
    --what hf_dataset --dataset-type real
```

### Scenario 3 — Combustion

```bash
benchmark download --dataset-root ./dataset --scenario combustion \
    --what hf_dataset --dataset-type numerical

benchmark download --dataset-root ./dataset --scenario combustion \
    --what hf_dataset --dataset-type real
```

After download, the dataset directory should look like:

```
dataset/
├── cylinder/hf_dataset/{numerical,real}/
├── controlled_cylinder/hf_dataset/{numerical,real}/
└── combustion/hf_dataset/{numerical,real}/
```

> **Storage estimate:** ~20 GB for all three scenarios (numerical + real).

---

## 🏋️ Step 2 — Pre-train on Simulation Data

This step trains a neural operator on **numerical (simulation) data only**,
producing a source model that encodes physical priors. The checkpoint from
this step is the starting point for all fine-tuning methods in Step 3.

We use **`1-cylinder × FNO`** as the running example throughout.
Replace `1-cylinder` / `fno` with any combination from the table below.

| Scenario | Config folder | Architecture choices |
|---|---|---|
| Cylinder flow | `configs/1-cylinder/` | `fno`, `cno`, `deeponet`, `transolver` |
| Controlled cylinder | `configs/2-controlled_cylinder/` | `fno`, `cno`, `deeponet`, `transolver` |
| Combustion | `configs/3-combustion/` | `fno`, `cno`, `deeponet`, `transolver` |

### Run pre-training

```bash
cd /path/to/PhysGuard

python -m physguard.pretrain \
    --config configs/1-cylinder/fno/1_pretrain.yaml
```

The config already sets `dataset_root: ./dataset/` and `results_path:
./results/001-cylinder/`. Training logs, tensorboard files, and periodic
checkpoints are written to:

```
results/001-cylinder/fno/fno_cylinder_pretrained/<timestamp>/
├── train.log
├── tb/
├── model_0080.pth
├── model_0160.pth
└── ...
```

**Note the checkpoint path** — you will need it in Step 3. It looks like:

```
./results/001-cylinder/fno/fno_cylinder_pretrained/2026-05-01_00-03-08/model_3840.pth
```

### Fill in the checkpoint path for Step 3

Open **all** `2_*.yaml` files for the scenario/architecture you just trained
and fill in `checkpoint_path`:

```yaml
# configs/1-cylinder/fno/2_physguard.yaml  (and 2_dft.yaml, 2_ewc.yaml, 2_l2sp.yaml)
checkpoint_path: ./results/001-cylinder/fno/fno_cylinder_pretrained/<timestamp>/model_XXXX.pth
```

---

## 🔬 Step 3 — Sim-to-Real Fine-Tuning

All five paradigms share the **same dataset, batch size, learning rate, and
number of update steps** so results are directly comparable.

Make sure `checkpoint_path` is filled in the YAML before running (see Step 2).

### Method A — PhysGuard (ours ★)

```bash
python -m physguard.finetune_physguard \
    --config configs/1-cylinder/fno/2_physguard.yaml
```

PhysGuard runs in two phases:

**Phase 1 (FIM SVD, run once):** Loads the pre-trained checkpoint, passes
`ns_max_samples` simulation batches through the model, computes the per-layer
empirical FIM via the Gram trick, and retains the top eigenvectors covering
`ns_variance_threshold` of Fisher variance. A per-layer summary is printed:

```
[layer fc.weight]  d=4096   N=200  -> k=18  (τ=0.90, ratio=18/200)
```

**Phase 2 (constrained fine-tuning):** Standard fine-tuning on real data with
gradients projected onto the identified null space at every step. The adapted
checkpoint is saved to:

```
results/001-cylinder/fno/fno_cylinder_physguard/<timestamp>/
```

Key hyperparameters (all configurable via CLI or YAML):

| Parameter | Meaning | Default |
|---|---|---|
| `ns_variance_threshold` | Fisher fraction `τ` for adaptive `k` selection per layer | `0.9` |
| `ns_n_components` | Hard cap on `k` | `200` |
| `ns_alpha` | Projection strength (`1.0` = full null-space, `0.0` = vanilla FT) | `1.0` |
| `ns_max_samples` | Simulation samples used to estimate the FIM | `200` |
| `ns_layer_wise` | Apply projection per-layer (recommended) | `True` |

### Method B — Direct Fine-Tuning (DFT)

```bash
python -m physguard.finetune_baselines \
    --config configs/1-cylinder/fno/2_dft.yaml
```

### Method C — Elastic Weight Consolidation (EWC)

```bash
python -m physguard.finetune_baselines \
    --config configs/1-cylinder/fno/2_ewc.yaml
```

### Method D — L2-SP Regularisation

```bash
python -m physguard.finetune_baselines \
    --config configs/1-cylinder/fno/2_l2sp.yaml
```

### Method E — Zero-shot (pretrained, no adaptation)

No training needed — directly evaluate the Step 2 checkpoint (see Step 4).

---

## 📊 Step 4 — Evaluation

```bash
python -m benchmark.eval \
    --config configs/1-cylinder/fno/2_physguard.yaml \
    --checkpoint_path results/001-cylinder/fno/fno_cylinder_physguard/<timestamp>/model_XXXX.pth
```

For zero-shot evaluation of the pretrained model:

```bash
python -m benchmark.eval \
    --config configs/1-cylinder/fno/1_pretrain.yaml \
    --checkpoint_path results/001-cylinder/fno/fno_cylinder_pretrained/<timestamp>/model_XXXX.pth
```

The script reports 9 RealPDEBench metrics. Per the manuscript, we focus on:
- **Rel L₂** — overall accuracy
- **Low-f / Mid-f / High-f RMSE** — spectral fidelity across frequency bands

Results are saved to `<exp_dir>/eval/eval.log` alongside qualitative plots.

### Summary table — all methods for one scenario

| Method | Config | Entry point |
|---|---|---|
| Zero-shot (pretrained) | `fno/1_pretrain.yaml` | `benchmark.eval` only |
| DFT | `fno/2_dft.yaml` | `physguard.finetune_baselines` → `benchmark.eval` |
| EWC | `fno/2_ewc.yaml` | `physguard.finetune_baselines` → `benchmark.eval` |
| L2-SP | `fno/2_l2sp.yaml` | `physguard.finetune_baselines` → `benchmark.eval` |
| **PhysGuard** ★ | `fno/2_physguard.yaml` | `physguard.finetune_physguard` → `benchmark.eval` |

---

## 📁 Output directory structure

Every run writes to:

```
results/<scenario>/<model>/<exp_name>_<paradigm>/<timestamp>/
├── args.txt          all CLI / YAML args (for reproducibility)
├── train.log         step-by-step training log
├── tb/               TensorBoard event files
├── model_<step>.pth  periodic checkpoints
├── projection.pt     (PhysGuard only) cached FIM SVD — reused across reruns
└── eval/             evaluation log + figures
```

`<paradigm>` is one of: `pretrained`, `dft`, `ewc`, `l2sp`, `physguard`.

---

## 🧩 Using PhysGuard in your own pipeline

Two classes integrate PhysGuard into any PyTorch training loop:

```python
from physguard import NullSpaceProjector, NullSpaceOptimizer

# Step A: estimate the FIM null space once, using simulation data
projector = NullSpaceProjector(variance_threshold=0.9, alpha=1.0)
projector.compute_projection(model, sim_dataloader, max_samples=200)

# Step B: wrap any optimizer — gradients are projected automatically
opt = NullSpaceOptimizer(
    torch.optim.Adam(model.parameters(), lr=1e-4),
    model, projector
)

for x, y in real_dataloader:
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()       # projects g → g - α·UUᵀg per layer, then steps
    opt.zero_grad()
```

The projector is architecture-agnostic and handles complex-valued spectral
weights (e.g. FNO) by splitting real/imaginary parts internally.
See [`physguard/projector.py`](physguard/projector.py) for full details.

---

## 🙏 Acknowledgements

The dataset, training pipeline, and baseline architectures are built on top
of [RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench)
(ICLR 2026 Oral). The null-space projection idea is inspired by
[GPM (ICLR 2021)](https://openreview.net/forum?id=3AOj0RCNC2) and
[AlphaEdit (ICLR 2025)](https://github.com/jianghoucheng/AlphaEdit), which
apply similar geometric protection mechanisms in continual learning and model
editing respectively.

## 📜 License

The PhysGuard additions (`physguard/`, `figures/`, top-level configs and
README) are released under the **MIT** license (`LICENSE`). The benchmark
code in `benchmark/` is governed by `LICENSE.RealPDEBench` (CC BY-NC 4.0).
