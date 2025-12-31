#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — PhysGuard end-to-end smoke test
# =============================================================================
# Walks through the complete pipeline from scratch for ALL THREE datasets:
#   cylinder  |  controlled_cylinder  |  combustion
#
# Each dataset exercises two phases:
#   Phase A) Pretrain on numerical simulation data     (~20 gradient steps)
#   Phase B) PhysGuard fine-tune on real experimental data (~20 gradient steps)
#
# Total expected runtime: ~5–10 min (first run includes Arrow index build)
#
# PREREQUISITES
#   - NVIDIA GPU with CUDA
#   - conda installed
#
# USAGE
#   bash smoke_test.sh                                      # GPU 0, direct HF
#   SMOKE_GPU=2 bash smoke_test.sh                          # GPU 2
#   HF_ENDPOINT=https://hf-mirror.com bash smoke_test.sh   # behind GFW
#   SMOKE_GPU=1 HF_ENDPOINT=https://hf-mirror.com bash smoke_test.sh
# =============================================================================

set -euo pipefail

# ── User-tuneable variables ───────────────────────────────────────────────────
SMOKE_GPU=${SMOKE_GPU:-0}
CONDA_ENV=${CONDA_ENV:-realpdebench}
DATA_ROOT=${DATA_ROOT:-./data/realpdebench}
# HF_ENDPOINT: leave unset for direct access; set to https://hf-mirror.com
#              if you are behind GFW.

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

step() { echo -e "\n${BOLD}${CYAN}━━━ $* ━━━${RESET}"; }
ok()   { echo -e "  ${GREEN}✓${RESET} $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
die()  { echo -e "\n${RED}ERROR:${RESET} $*" >&2; exit 1; }

# ── Helper: find latest model checkpoint under a directory ───────────────────
# Filenames are model_<step>.pth; pick the highest step number.
latest_checkpoint() {
    find "$1" -name "model_*.pth" 2>/dev/null \
        | awk -F'[_.]' '{print $(NF-1), $0}' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

# ── Helper: patch checkpoint_path + gpu into a temp copy of a YAML config ────
patch_yaml() {          # patch_yaml <src> <ckpt> <gpu> <dst>
    local src="$1" ckpt="$2" gpu="$3" dst="$4"
    conda run -n "${CONDA_ENV}" python -c "
import yaml
with open('${src}') as f:
    cfg = yaml.safe_load(f)
cfg['gpu'] = ${gpu}
cfg['checkpoint_path'] = '${ckpt}'
with open('${dst}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"
}

# ── Helper: run pretrain (patches GPU into temp YAML, then trains) ────────────
run_pretrain() {        # run_pretrain <config_path>
    local config="$1"
    local tmp
    tmp=$(mktemp /tmp/smoke_XXXXXX.yaml)
    conda run -n "${CONDA_ENV}" python -c "
import yaml
with open('${config}') as f:
    cfg = yaml.safe_load(f)
cfg['gpu'] = ${SMOKE_GPU}
with open('${tmp}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"
    conda run -n "${CONDA_ENV}" \
        python -m realpdebench.train_gpus --config "${tmp}"
    rm -f "${tmp}"
}

# =============================================================================
# STEP 1 — Environment setup
# =============================================================================
step "STEP 1 · Create / verify conda environment"

if conda env list 2>/dev/null | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    ok "conda env '${CONDA_ENV}' already exists — skipping creation"
else
    echo "  Creating conda env '${CONDA_ENV}' with Python 3.10 ..."
    conda create -y -n "${CONDA_ENV}" python=3.10
    ok "conda env created"
fi

# Check for PyTorch; warn if missing (user must install the right CUDA build)
conda run -n "${CONDA_ENV}" python -c "import torch" 2>/dev/null \
    && ok "torch is available" \
    || {
        warn "torch not found in '${CONDA_ENV}'."
        warn "Install the CUDA build matching your driver — see https://pytorch.org/get-started"
        warn "Example (CUDA 12.8):  pip install torch --index-url https://download.pytorch.org/whl/cu128"
        die  "Please install torch first, then re-run this script."
    }

echo "  pip install -e . (editable, into '${CONDA_ENV}') ..."
conda run -n "${CONDA_ENV}" pip install -q -e .
ok "package installed"

# =============================================================================
# STEP 2 — Download datasets
# =============================================================================
step "STEP 2 · Download datasets from Hugging Face"
echo "  Repo   : AI4Science-WestlakeU/RealPDEBench"
echo "  Splits : numerical (pretrain)  +  real (fine-tune)"
echo "  Target : ${DATA_ROOT}"
[ -n "${HF_ENDPOINT:-}" ] \
    && echo "  Mirror : ${HF_ENDPOINT}" \
    || echo "  Mirror : direct (set HF_ENDPOINT=https://hf-mirror.com if behind GFW)"

DOWNLOAD_ARGS=(
    python -m realpdebench download
    --dataset-root "${DATA_ROOT}"
    --scenario cylinder
    --scenario controlled_cylinder
    --scenario combustion
    --what hf_dataset
    --dataset-type numerical
    --dataset-type real
)
[ -n "${HF_ENDPOINT:-}" ] && DOWNLOAD_ARGS+=(--endpoint "${HF_ENDPOINT}")

conda run -n "${CONDA_ENV}" "${DOWNLOAD_ARGS[@]}"
ok "all three datasets downloaded to ${DATA_ROOT}"

# =============================================================================
# STEP 3 — Cylinder pretrain
# =============================================================================
step "STEP 3 · Cylinder — pretrain  (numerical, 20 steps, GPU ${SMOKE_GPU})"
run_pretrain configs/smoke/1_cylinder_pretrain.yaml

CKPT_CYL=$(latest_checkpoint ./results/smoke/001-cylinder)
[ -z "${CKPT_CYL}" ] && die "No checkpoint found under ./results/smoke/001-cylinder"
ok "Checkpoint: ${CKPT_CYL}"

# =============================================================================
# STEP 4 — Cylinder PhysGuard fine-tune
# =============================================================================
step "STEP 4 · Cylinder — PhysGuard fine-tune  (real, 20 steps, GPU ${SMOKE_GPU})"
TMP_CYL=$(mktemp /tmp/smoke_cyl_pg_XXXXXX.yaml)
patch_yaml configs/smoke/2_cylinder_physguard.yaml "${CKPT_CYL}" "${SMOKE_GPU}" "${TMP_CYL}"
conda run -n "${CONDA_ENV}" \
    python -m realpdebench.train_gpus \
        --config "${TMP_CYL}" \
        --is_finetune
rm -f "${TMP_CYL}"
ok "Cylinder smoke passed ✓"

# =============================================================================
# STEP 5 — Controlled Cylinder pretrain
# =============================================================================
step "STEP 5 · Controlled Cylinder — pretrain  (numerical, 20 steps, GPU ${SMOKE_GPU})"
run_pretrain configs/smoke/3_controlled_cylinder_pretrain.yaml

CKPT_CTRL=$(latest_checkpoint ./results/smoke/002-controlled_cylinder)
[ -z "${CKPT_CTRL}" ] && die "No checkpoint found under ./results/smoke/002-controlled_cylinder"
ok "Checkpoint: ${CKPT_CTRL}"

# =============================================================================
# STEP 6 — Controlled Cylinder PhysGuard fine-tune
# =============================================================================
step "STEP 6 · Controlled Cylinder — PhysGuard fine-tune  (real, 20 steps, GPU ${SMOKE_GPU})"
TMP_CTRL=$(mktemp /tmp/smoke_ctrl_pg_XXXXXX.yaml)
patch_yaml configs/smoke/4_controlled_cylinder_physguard.yaml "${CKPT_CTRL}" "${SMOKE_GPU}" "${TMP_CTRL}"
conda run -n "${CONDA_ENV}" \
    python -m realpdebench.train_gpus \
        --config "${TMP_CTRL}" \
        --is_finetune
rm -f "${TMP_CTRL}"
ok "Controlled Cylinder smoke passed ✓"

# =============================================================================
# STEP 7 — Combustion pretrain
# =============================================================================
step "STEP 7 · Combustion — pretrain  (numerical, 20 steps, GPU ${SMOKE_GPU})"
run_pretrain configs/smoke/5_combustion_pretrain.yaml

CKPT_CMB=$(latest_checkpoint ./results/smoke/003-combustion)
[ -z "${CKPT_CMB}" ] && die "No checkpoint found under ./results/smoke/003-combustion"
ok "Checkpoint: ${CKPT_CMB}"

# =============================================================================
# STEP 8 — Combustion PhysGuard fine-tune
# =============================================================================
step "STEP 8 · Combustion — PhysGuard fine-tune  (real, 20 steps, GPU ${SMOKE_GPU})"
TMP_CMB=$(mktemp /tmp/smoke_cmb_pg_XXXXXX.yaml)
patch_yaml configs/smoke/6_combustion_physguard.yaml "${CKPT_CMB}" "${SMOKE_GPU}" "${TMP_CMB}"
conda run -n "${CONDA_ENV}" \
    python -m realpdebench.train_gpus \
        --config "${TMP_CMB}" \
        --is_finetune
rm -f "${TMP_CMB}"
ok "Combustion smoke passed ✓"

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${BOLD}${GREEN}All 8 smoke steps passed.${RESET}"
echo "Smoke checkpoints saved under ./results/smoke/"
echo ""
echo "Next steps — full training:"
echo "  pretrain :  python -m realpdebench.train_gpus --config configs/1-cylinder/fno/1_pretrain.yaml"
echo "  PhysGuard:  python -m realpdebench.train_gpus --config configs/1-cylinder/fno/2_physguard.yaml --is_finetune"
