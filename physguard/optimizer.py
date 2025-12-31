"""
Null Space Optimizer Wrapper.

Wraps a standard PyTorch optimizer and automatically projects gradients
into the null space before each optimizer step.
"""

import torch
import torch.nn as nn
from typing import Optional
from physguard.projector import NullSpaceProjector


class NullSpaceOptimizer:
    """
    A wrapper around a standard optimizer that applies null-space gradient
    projection before each step.

    Usage:
        projector = NullSpaceProjector(...)
        projector.compute_projection(model, sim_dataloader, ...)

        base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer = NullSpaceOptimizer(base_optimizer, model, projector)

        # Training loop
        loss.backward()
        optimizer.step()  # automatically projects gradients first
        optimizer.zero_grad()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        projector: NullSpaceProjector,
    ):
        self.optimizer = optimizer
        self.model = model
        self.projector = projector

    def step(self):
        """Project gradients into null space, then step the base optimizer."""
        self.projector.project_gradients(self.model)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
