# PhysGuard environment setup script for Windows (PowerShell).
# Automatically detects CUDA driver version and installs compatible torch.
#
# Usage:
#   conda env create -f environment.yml
#   conda activate physguard
#   .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> Detecting CUDA driver version..."

# --- A: search nvidia-smi in PATH and common Windows install locations ---
$CUDA_VER = $null
$nvidiaSmiExe = $null

$candidates = @(
    "nvidia-smi",   # already in PATH
    "$env:SystemRoot\System32\nvidia-smi.exe",
    "C:\Windows\System32\nvidia-smi.exe",
    "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    "C:\Program Files\NVIDIA Corporation\NvSMI\nvidia-smi.exe"
)

foreach ($candidate in $candidates) {
    try {
        $out = & $candidate 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $nvidiaSmiExe = $candidate
            break
        }
    } catch {}
}

if ($nvidiaSmiExe) {
    $out = & $nvidiaSmiExe 2>$null | Select-String "CUDA Version: (\d+\.\d+)"
    if ($out) {
        $CUDA_VER = $out.Matches[0].Groups[1].Value
    }
}

# --- B: if still not found, ask the user ---
if (-not $CUDA_VER) {
    Write-Host ""
    Write-Host "    Could not detect CUDA version automatically (nvidia-smi not found)."
    Write-Host "    Please run 'nvidia-smi' manually in another window to find your CUDA version,"
    Write-Host "    then enter it here (e.g. 12.4), or press Enter to install CPU-only torch."
    $input = Read-Host "    Your CUDA version"
    $input = $input.Trim()
    if ($input -match "^\d+\.\d+$") {
        $CUDA_VER = $input
    }
}

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
