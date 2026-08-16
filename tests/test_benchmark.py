"""Tests for deterministic benchmark-result selection."""

from pilotnet.tools.benchmark import BenchmarkResult, select_best_configuration


def test_select_best_configuration_uses_highest_throughput() -> None:
    results = [
        BenchmarkResult(batch_size=32, workers=2, samples_per_second=100.0),
        BenchmarkResult(batch_size=64, workers=0, samples_per_second=125.0),
        BenchmarkResult(batch_size=128, workers=4, samples_per_second=120.0),
    ]

    selected = select_best_configuration(results)

    assert selected == results[1]


def test_select_best_configuration_prefers_smaller_resources_for_a_tie() -> None:
    results = [
        BenchmarkResult(batch_size=128, workers=4, samples_per_second=100.0),
        BenchmarkResult(batch_size=64, workers=2, samples_per_second=100.0),
    ]

    assert select_best_configuration(results) == results[1]
