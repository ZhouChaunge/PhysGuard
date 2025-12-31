#!/bin/bash
# Eval CNO + DeepONet on cylinder dataset (real test data)
# Minimal plot mode: each (sample, time_step) saved as individual PDF/PNG
set -e

cd ./RealPDEBench

GPU=1

echo "========== CNO Cylinder =========="

# CNO pretrained (best=4300)
echo "[CNO] Pretrained..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/cno.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/cno/cno_cylinder_pretrained/2026-03-11_09-49-04/model_4300.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

# CNO DFT (best=1120)
echo "[CNO] DFT..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/cno.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/cno/cno_cylinder_dft/2026-03-13_15-20-44/model_1120.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

# CNO EWC (best=1520)
echo "[CNO] EWC..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/cno.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/cno/cno_cylinder_ewcft/2026-03-13_15-16-11/model_1520.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

# CNO L2-SP (best=1120)
echo "[CNO] L2-SP..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/cno.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/cno/cno_cylinder_l2ft/lam0.0001/2026-03-13_15-16-15/model_1120.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

# CNO PhysGuard (best=720)
echo "[CNO] PhysGuard..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/cno.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/cno/cno_cylinder_nsft/2026-03-14_06-37-42/model_0720.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

echo ""
echo "========== DeepONet Cylinder =========="

# DeepONet pretrained (best=100)
echo "[DeepONet] Pretrained..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/deeponet.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/deeponet/deeponet_cylinder_pretrained/2026-03-12_11-16-03/model_0100.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

# DeepONet DFT (best=560)
echo "[DeepONet] DFT..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/deeponet.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/deeponet/deeponet_cylinder_dft/2026-03-13_22-22-57/model_0560.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

# DeepONet EWC (best=2720)
echo "[DeepONet] EWC..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/deeponet.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/deeponet/deeponet_cylinder_ewcft/lam1.0/2026-03-14_06-34-38/model_2720.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

# DeepONet L2-SP (best=2560)
echo "[DeepONet] L2-SP..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/deeponet.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/deeponet/deeponet_cylinder_l2ft/lam0.0001/2026-03-14_06-34-38/model_2560.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

# DeepONet PhysGuard (best=80)
echo "[DeepONet] PhysGuard..."
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/deeponet.yaml \
  --gpu $GPU \
  --checkpoint_path results/001-cylinder/deeponet/deeponet_cylinder_nsft/2026-03-13_00-38-45/model_0080.pth \
  --test_data_type real --train_data_type numerical \
  --use_hf_dataset --N_plot 4 --test_batch_size 4 \
  --minimal_plot

echo ""
echo "========== ALL DONE =========="
