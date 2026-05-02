# PhysGuard environment setup script for Windows (PowerShell).
# Automatically detects CUDA driver version and installs compatible torch.
#
# Usage:
#   conda env create -f environment.yml
#   conda activate physguard
#   .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> Detecting CUDA driver version..."

$CUDA_VER = $null
try {
    $nvidiaSmi = nvidia-smi 2>$null | Select-String "CUDA Version: (\d+\.\d+)"
    if ($nvidiaSmi) {
        $CUDA_VER = $nvidiaSmi.Matches[0].Groups[1].Value
    }
} catch {}

if (-not $CUDA_VER) {
    Write-Host "    No GPU detected. Installing CPU-only torch."
    $TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
    $TORCH_VER   = "torch torchvision"
} else {
    $parts = $CUDA_VER -split "\."
    $MAJOR = [int]$parts[0]
    $MINOR = [int]$parts[1]
    Write-Host "    Driver supports CUDA <= $CUDA_VER"

    if ($MAJOR -ge 13) {
        $TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
        $TORCH_VER   = "torch torchvision"
    } elseif ($MAJOR -eq 12 -and $MINOR -ge 4) {
        $TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
        $TORCH_VER   = "torch==2.5.1+cu124 torchvision==0.20.1+cu124"
    } elseif ($MAJOR -eq 12 -and $MINOR -ge 1) {
        $TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
        $TORCH_VER   = "torch==2.3.1+cu121 torchvision==0.18.1+cu121"
    } elseif ($MAJOR -eq 11 -and $MINOR -ge 8) {
        $TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
        $TORCH_VER   = "torch==2.3.1+cu118 torchvision==0.18.1+cu118"
    } else {
        Write-Host "    WARNING: CUDA $CUDA_VER is too old (< 11.8). Installing CPU-only torch."
        $TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
        $TORCH_VER   = "torch torchvision"
    }
}

Write-Host "==> Installing: $TORCH_VER"
Write-Host "    from: $TORCH_INDEX"
pip install $TORCH_VER.Split(" ") --index-url $TORCH_INDEX

Write-Host "==> Installing project (editable)..."
pip install -e .

Write-Host ""
Write-Host "==> Verifying..."
python -c @"
import torch
print(f'torch:          {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:            {torch.cuda.get_device_name(0)}')
    print(f'CUDA:           {torch.version.cuda}')
"@

Write-Host ""
Write-Host "Done! Environment is ready."
