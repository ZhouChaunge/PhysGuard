"""
PhysGuard · Step 1 — Pre-train a neural-operator on simulated data.

Usage:
    python -m physguard.pretrain --config configs/<scenario>/<model>/1_pretrain.yaml

The simulated training data is what makes the resulting weights serve as
the source model for *all* downstream finetuning methods (DFT, L2-SP, EWC,
PhysGuard).  This entry point is intentionally a thin wrapper around the
shared training engine so that the per-step training logic stays identical
to the baseline finetuning runs.
"""
from __future__ import annotations

from physguard._engine import launch


if __name__ == "__main__":
    launch(mode="pretrain")
