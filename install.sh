#!/bin/bash
# PhysGuard environment setup script.
# Automatically detects CUDA driver version and installs compatible torch.
#
# Usage:
#   conda env create -f environment.yml [-p /path/to/env]
#   conda activate physguard  (or: conda activate /path/to/env)
#   bash install.sh

set -e

echo "==> Detecting CUDA driver version..."

# Get driver-supported CUDA version from nvidia-smi
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)

if [ -z "$CUDA_VER" ]; then
    echo "    No GPU detected. Installing CPU-only torch."
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_VER="torch torchvision"
else
    MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    echo "    Driver supports CUDA <= ${CUDA_VER}"

    if [ "$MAJOR" -ge 13 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu130"
        TORCH_VER="torch torchvision"
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 4 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_VER="torch==2.5.1+cu124 torchvision==0.20.1+cu124"
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 1 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_VER="torch==2.3.1+cu121 torchvision==0.18.1+cu121"
    elif [ "$MAJOR" -eq 11 ] && [ "$MINOR" -ge 8 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_VER="torch==2.3.1+cu118 torchvision==0.18.1+cu118"
    else
        echo "    WARNING: CUDA ${CUDA_VER} is too old (< 11.8). Installing CPU-only torch."
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_VER="torch torchvision"
    fi
fi

echo "==> Installing: ${TORCH_VER}"
echo "    from: ${TORCH_INDEX}"
pip install ${TORCH_VER} --index-url "${TORCH_INDEX}"

echo "==> Installing project (editable)..."
pip install -e .

echo ""
echo "==> Verifying..."
python -c "
import torch
print(f'torch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:          {torch.cuda.get_device_name(0)}')
    print(f'CUDA:         {torch.version.cuda}')
"
echo ""
echo "Done! Environment is ready."
