"""
PhysGuard · Step 2 (baselines) — Fine-tune via DFT / L2-SP / EWC.

Usage:
    python -m physguard.finetune_baselines --config configs/<scenario>/<model>/2_dft.yaml
    python -m physguard.finetune_baselines --config configs/<scenario>/<model>/2_l2sp.yaml
    python -m physguard.finetune_baselines --config configs/<scenario>/<model>/2_ewc.yaml

The specific baseline is selected by the ``reg_type`` field inside the YAML:

    reg_type: "none"  →  Direct Fine-Tuning  (DFT)
    reg_type: "l2"    →  L2-SP regularisation
    reg_type: "ewc"   →  Elastic Weight Consolidation

These are the comparison methods reported alongside our PhysGuard results
in the paper.  PhysGuard itself has its own dedicated entry point in
``physguard.finetune_physguard``.
"""
from __future__ import annotations

from physguard._engine import launch


if __name__ == "__main__":
    launch(mode="finetune")
