"""Small, reusable supervised training routines."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _metrics(total_squared_error: float, total_absolute_error: float, count: int) -> dict[str, float]:
    return {"mse": total_squared_error / count, "mae": total_absolute_error / count}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Optimize mean squared steering error for one epoch."""
    model.train()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        predictions = model(images)
        errors = predictions - targets
        loss = errors.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        squared_error += errors.detach().square().sum().item()
        absolute_error += errors.detach().abs().sum().item()
        count += targets.numel()
    return _metrics(squared_error, absolute_error, count)


@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    """Measure mean squared and absolute steering error."""
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    for images, targets in dataloader:
        errors = model(images.to(device)) - targets.to(device)
        squared_error += errors.square().sum().item()
        absolute_error += errors.abs().sum().item()
        count += targets.numel()
    return _metrics(squared_error, absolute_error, count)
