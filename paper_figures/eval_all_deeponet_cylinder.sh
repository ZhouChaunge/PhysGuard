#!/bin/bash
# Evaluate all DeepONet models on cylinder flow (real test data) with flow field plots.
# Each model's eval output (metrics + flow field images) goes to {checkpoint_dir}/eval/

set -e
cd ./RealPDEBench

PYTHON=python
CONFIG=realpdebench/configs/cylinder/deeponet.yaml
BASE=./results/001-cylinder/deeponet

# Method -> (checkpoint_path, exp_suffix)
declare -A CKPTS
CKPTS[pretrained]="${BASE}/deeponet_cylinder_pretrained/2026-03-12_11-16-03/model_0100.pth"
CKPTS[dft]="${BASE}/deeponet_cylinder_dft/2026-03-13_22-22-57/model_0560.pth"
CKPTS[ewc]="${BASE}/deeponet_cylinder_ewcft/lam1.0/2026-03-14_06-34-38/model_2720.pth"
CKPTS[l2sp]="${BASE}/deeponet_cylinder_l2ft/lam0.0001/2026-03-14_06-34-38/model_2560.pth"
CKPTS[physguard]="${BASE}/deeponet_cylinder_nsft/2026-03-13_00-38-45/model_0080.pth"

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
        --N_plot_probe 12 \
        --gpu 1 \
        --exp_suffix "$method"
    echo ""
done

echo "All evaluations complete!"
echo "Results saved under each checkpoint directory's eval_<method>/ folder."
