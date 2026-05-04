# PhysGuard

> **Select your operating system below to expand the complete step-by-step guide:**
>
> | | OS | Jump to |
> |---|---|---|
> | 🪟 | Windows (PowerShell) | [▶ Windows guide](#-quick-start--select-your-os) |
> | 🐧 | Linux (Bash) | [▶ Linux guide](#-quick-start--select-your-os) |
> | 🍎 | macOS (Zsh/Bash) | [▶ macOS guide](#-quick-start--select-your-os) |

---

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

## 🚀 Quick Start — Select Your OS

> **Requirements:** Python ≥ 3.10, CUDA ≥ 11.8, one GPU (RTX 4090 / A40 / A100 is sufficient).

Click your operating system to expand the complete step-by-step guide:

---

<details>
<summary><b>🪟 Windows (PowerShell) — Complete Guide (Steps 0–4)</b></summary>

<br>

> All commands use PowerShell — open **Anaconda PowerShell Prompt** from the Start Menu.

**Prerequisites:**
1. Install **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** (64-bit, Python 3.11+)
2. Install **[Git for Windows](https://git-scm.com/download/win)**
3. Verify your NVIDIA driver: run `watch -n 0.1 nvidia-smi` in any PowerShell window

---

### Step 0 — Setup

```powershell
# Clone
git clone https://github.com/<anonymous>/PhysGuard.git
cd PhysGuard

# Create and activate env
conda env create -f environment.yml
conda activate physguard

# Install PyTorch (auto-detects CUDA)
powershell -ExecutionPolicy Bypass -File install.ps1

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import physguard; import benchmark; print('OK')"
```

---

### Step 1 — Download Data (Cylinder example)

```powershell
# If behind GFW, set these first:
$env:HF_HUB_DISABLE_XET = "1"
# Then append  --endpoint https://hf-mirror.com  to each command below

benchmark download --dataset-root .\dataset --scenario cylinder --what hf_dataset --dataset-type numerical
benchmark download --dataset-root .\dataset --scenario cylinder --what hf_dataset --dataset-type real
```

---

### Step 2 — Pre-train on Simulation Data

```powershell
python -m physguard.pretrain --config configs\1-cylinder\fno\1_pretrain.yaml
```

Find and note the checkpoint path:
```powershell
Get-ChildItem .\results\001-cylinder\fno\ -Recurse -Filter "*.pth" | Sort-Object LastWriteTime | Select-Object -Last 1 -ExpandProperty FullName
```

Edit each `2_*.yaml` and set `checkpoint_path` to that path.

---

### Step 3 — Sim-to-Real Fine-Tuning

```powershell
# PhysGuard (ours)
python -m physguard.finetune_physguard --config configs\1-cylinder\fno\2_physguard.yaml

# Baselines (pick one)
python -m physguard.finetune_baselines --config configs\1-cylinder\fno\2_dft.yaml
python -m physguard.finetune_baselines --config configs\1-cylinder\fno\2_ewc.yaml
python -m physguard.finetune_baselines --config configs\1-cylinder\fno\2_l2sp.yaml
```

---

### Step 4 — Evaluation

```powershell
python -m benchmark.eval --config configs\1-cylinder\fno\2_physguard.yaml --checkpoint_path results\001-cylinder\fno\fno_cylinder_physguard\<timestamp>\model_XXXX.pth
```

Results are saved to `<exp_dir>\eval\eval.log`.

</details>

---

<details>
<summary><b>🐧 Linux (Bash) — Complete Guide (Steps 0–4)</b></summary>

<br>

**Prerequisites:**

```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash && exec bash

# Verify NVIDIA driver
watch -n 0.1 nvidia-smi
```

---

### Step 0 — Setup

```bash
git clone https://github.com/<anonymous>/PhysGuard.git
cd PhysGuard

conda env create -f environment.yml
conda activate physguard

bash install.sh

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import physguard; import benchmark; print('OK')"
```

---

### Step 1 — Download Data (Cylinder example)

```bash
# If behind GFW:
export HF_HUB_DISABLE_XET=1
# Then append  --endpoint https://hf-mirror.com  to each command below

benchmark download --dataset-root ./dataset --scenario cylinder \
    --what hf_dataset --dataset-type numerical

benchmark download --dataset-root ./dataset --scenario cylinder \
    --what hf_dataset --dataset-type real
```

---

### Step 2 — Pre-train on Simulation Data

```bash
python -m physguard.pretrain \
    --config configs/1-cylinder/fno/1_pretrain.yaml
```

Find the checkpoint path:
```bash
find ./results/001-cylinder/fno/ -name "*.pth" | sort | tail -1
```

Edit each `2_*.yaml` and set `checkpoint_path` to that path.

---

### Step 3 — Sim-to-Real Fine-Tuning

```bash
# PhysGuard (ours)
python -m physguard.finetune_physguard \
    --config configs/1-cylinder/fno/2_physguard.yaml

# Baselines (pick one)
python -m physguard.finetune_baselines --config configs/1-cylinder/fno/2_dft.yaml
python -m physguard.finetune_baselines --config configs/1-cylinder/fno/2_ewc.yaml
python -m physguard.finetune_baselines --config configs/1-cylinder/fno/2_l2sp.yaml
```

---

### Step 4 — Evaluation

```bash
python -m benchmark.eval \
    --config configs/1-cylinder/fno/2_physguard.yaml \
    --checkpoint_path results/001-cylinder/fno/fno_cylinder_physguard/<timestamp>/model_XXXX.pth
```

Results are saved to `<exp_dir>/eval/eval.log` alongside qualitative plots.

</details>

---

<details>
<summary><b>🍎 macOS (Zsh/Bash) — Complete Guide (Steps 0–4)</b></summary>

<br>

> macOS does **not** support NVIDIA CUDA. Recommended for data download, code exploration, and light evaluation only. For full training, use a Linux/Windows GPU machine.

**Prerequisites:**

```zsh
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Miniconda and Git
brew install --cask miniconda
brew install git
conda init zsh && exec zsh
```

---

### Step 0 — Setup

```zsh
git clone https://github.com/<anonymous>/PhysGuard.git
cd PhysGuard

conda env create -f environment.yml
conda activate physguard

bash install.sh   # installs CPU/MPS PyTorch

python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
python -c "import physguard; import benchmark; print('OK')"
```

---

### Step 1 — Download Data (Cylinder example)

```zsh
# If behind GFW:
export HF_HUB_DISABLE_XET=1
# Then append  --endpoint https://hf-mirror.com  to each command below

benchmark download --dataset-root ./dataset --scenario cylinder \
    --what hf_dataset --dataset-type numerical

benchmark download --dataset-root ./dataset --scenario cylinder \
    --what hf_dataset --dataset-type real
```

---

### Steps 2–4

Commands are identical to the **🐧 Linux** guide above — `\` line continuation
works in both Zsh and Bash on macOS.

</details>

---

## 📊 Methods Summary

| Method | Config | Entry point |
|---|---|---|
| Zero-shot (pretrained) | `fno/1_pretrain.yaml` | `benchmark.eval` only |
| DFT | `fno/2_dft.yaml` | `physguard.finetune_baselines` → `benchmark.eval` |
| EWC | `fno/2_ewc.yaml` | `physguard.finetune_baselines` → `benchmark.eval` |
| L2-SP | `fno/2_l2sp.yaml` | `physguard.finetune_baselines` → `benchmark.eval` |
| **PhysGuard** ★ | `fno/2_physguard.yaml` | `physguard.finetune_physguard` → `benchmark.eval` |

The script reports 9 RealPDEBench metrics. Per the manuscript, we focus on:
- **Rel L₂** — overall accuracy
- **Low-f / Mid-f / High-f RMSE** — spectral fidelity across frequency bands

---

## 📁 Output Directory Structure

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

## 🧩 Using PhysGuard in Your Own Pipeline

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
