"""Measure PilotNet training throughput for DataLoader configurations."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pilotnet.data import DrivingDataset, PreprocessConfig
from pilotnet.models import PilotNet


@dataclass(frozen=True)
class BenchmarkResult:
    """A successfully measured DataLoader configuration."""

    batch_size: int
    workers: int
    samples_per_second: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def select_best_configuration(results: list[BenchmarkResult]) -> BenchmarkResult:
    """Choose maximum throughput, preferring lower resource use for ties."""
    if not results:
        raise ValueError("No successful benchmark configurations.")
    return min(results, key=lambda result: (-result.samples_per_second, result.batch_size, result.workers))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_configurations(
    csv_path: str | Path,
    *,
    preprocess: PreprocessConfig,
    device: torch.device,
    batch_sizes: list[int],
    workers: list[int],
    steps: int = 20,
    lr: float = 1e-4,
) -> tuple[list[BenchmarkResult], list[dict[str, int | str]]]:
    """Run real forward, backward, and optimizer steps for each candidate."""
    if steps < 1:
        raise ValueError("steps must be positive.")
    dataset = DrivingDataset(csv_path, preprocess=preprocess)
    results: list[BenchmarkResult] = []
    failures: list[dict[str, int | str]] = []
    for batch_size in batch_sizes:
        for worker_count in workers:
            try:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=worker_count,
                    shuffle=True,
                    pin_memory=device.type == "cuda",
                )
                model = PilotNet().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                batches = cycle(loader)
                _synchronize(device)
                started = time.perf_counter()
                samples = 0
                for _ in range(steps):
                    images, targets = next(batches)
                    predictions = model(images.to(device))
                    loss = (predictions - targets.to(device)).square().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    samples += targets.numel()
                _synchronize(device)
                elapsed = time.perf_counter() - started
                results.append(BenchmarkResult(batch_size, worker_count, samples / elapsed))
            except torch.cuda.OutOfMemoryError:
                failures.append({"batch_size": batch_size, "workers": worker_count, "error": "cuda_oom"})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return results, failures
