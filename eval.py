"""Evaluate a PilotNet checkpoint against a CSV driving log."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from pilotnet.data import DrivingDataset
from pilotnet.engine import evaluate
from pilotnet.models import PilotNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = PilotNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(DrivingDataset(args.csv), batch_size=args.batch_size, num_workers=args.workers)
    print(json.dumps(evaluate(model, loader, device), indent=2))


if __name__ == "__main__":
    main()
