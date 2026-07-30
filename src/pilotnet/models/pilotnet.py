"""PilotNet architecture from Bojarski et al. (2016)."""

from __future__ import annotations

import torch
from torch import nn


class PilotNet(nn.Module):
    """Map a cropped front-camera image to one steering-angle prediction.

    The network uses the five convolutional and four fully connected layers
    reported in the PilotNet paper. Inputs must be RGB tensors in ``[0, 1]``
    with shape ``(batch, 3, 66, 200)``; normalization is part of the model.
    """

    image_size = (66, 200)

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ELU(),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return steering predictions with shape ``(batch,)``."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, *self.image_size):
            raise ValueError(
                "Expected images with shape (batch, 3, 66, 200), "
                f"received {tuple(images.shape)}."
            )
        features = self.features(images - 0.5)
        return self.regressor(features).squeeze(-1)
