#!/bin/bash
# Evaluate all FNO models on combustion (real test data).
# Methods: pretrained, dft, ewc, l2sp, physguard
set -e
cd ./RealPDEBench

PYTHON=python
CONFIG=realpdebench/configs/combustion/fno.yaml
BASE=./results/003-combustion/fno

declare -A CKPTS
CKPTS[pretrained]="${BASE}/fno_combustion_pretrained/2026-03-20_11-54-28/model_1000.pth"
CKPTS[dft]="${BASE}/fno_combustion_dft/2026-03-22_12-19-34/model_4000.pth"
CKPTS[ewc]="${BASE}/fno_combustion_ewcft/2026-03-24_21-54-32/model_3920.pth"
CKPTS[l2sp]="${BASE}/fno_combustion_l2ft/2026-03-26_10-29-14/model_4000.pth"
CKPTS[physguard]="${BASE}/fno_combustion_nsft/2026-03-22_13-13-42/model_3840.pth"

for method in pretrained dft ewc l2sp physguard; do
    ckpt="${CKPTS[$method]}"
    echo "============================================"
    echo "Evaluating: $method"
    echo "Checkpoint: $ckpt"
    echo "============================================"
    $PYTHON -m realpdebench.eval \
        --config $CONFIG \
        --checkpoint_path "$ckpt" \
        --test_data_type real \
        --use_hf_dataset \
        --N_plot 3 \
        --test_batch_size 16 \
        --gpu 2 \
        --exp_suffix "$method"
    echo "Done: $method"
    echo ""
done

echo "ALL DONE"
