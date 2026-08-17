"""ImageNet-pretrained ResNet encoder with causal GRU steering head."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class TemporalResNetGRU(nn.Module):
    """Predict steering from a causal RGB image sequence and speed history."""

    def __init__(self, *, pretrained: bool = True, hidden_size: int = 256) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.encoder = resnet34(weights=weights)
        self.encoder.fc = nn.Identity()
        self.projection = nn.Sequential(nn.Linear(513, hidden_size), nn.ReLU())
        self.temporal = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.regressor = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, images: torch.Tensor, speeds: torch.Tensor) -> torch.Tensor:
        """Return one steering prediction for each ``[batch, time]`` sequence."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape (batch, time, 3, height, width).")
        if speeds.shape != images.shape[:2]:
            raise ValueError("speeds must have shape (batch, time) matching images.")
        batch, steps, channels, height, width = images.shape
        encoded = self.encoder(images.reshape(batch * steps, channels, height, width))
        encoded = encoded.reshape(batch, steps, -1)
        inputs = torch.cat((encoded, speeds.unsqueeze(-1)), dim=-1)
        outputs, _ = self.temporal(self.projection(inputs))
        return self.regressor(outputs[:, -1]).squeeze(-1)
