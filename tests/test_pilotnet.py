"""Architecture tests for PilotNet."""

import pytest
import torch

from pilotnet.models import PilotNet


def test_pilotnet_returns_one_steering_value_per_image() -> None:
    model = PilotNet()

    prediction = model(torch.rand(2, 3, 66, 200))

    assert prediction.shape == (2,)
    assert sum(parameter.numel() for parameter in model.parameters()) == 252_219


def test_pilotnet_rejects_an_unexpected_image_shape() -> None:
    with pytest.raises(ValueError, match="66, 200"):
        PilotNet()(torch.rand(1, 3, 64, 200))
