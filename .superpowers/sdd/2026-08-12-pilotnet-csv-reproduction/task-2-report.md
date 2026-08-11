# Task 2 Report: Balanced Sampling And Deterministic Run Metadata

## Delivered

- Added `build_balanced_sampler` with inverse-frequency steering-bin weights,
  replacement sampling, optional Torch generator support, bin validation, and
  uniform handling for identical targets.
- Added reproducibility utilities that seed Python, NumPy, and Torch, configure
  Torch deterministic algorithms, and build JSON-serializable run manifests.
- Added NumPy to `pyproject.toml` and `requirements.txt`.
- Added training CLI options for preprocessing crop fractions, side-camera
  correction, balance bins, deterministic mode, seed, artifact/output directory,
  and worker count.
- Training now creates a balanced sampled train loader, leaves validation
  unshuffled, writes `run_manifest.json` before epoch one, and embeds that
  manifest in `best.pt`.
- Evaluation now reconstructs preprocessing from checkpoint metadata and rejects
  checkpoints without preprocessing metadata.

## TDD Evidence

- Added sampler and reproducibility tests first; their initial focused run failed
  during collection because `pilotnet.data.sampling` and `pilotnet.utils` did not
  exist.
- Added train/eval CLI integration tests next; their initial run failed because
  training lacked the new CLI options and evaluation accepted checkpoints without
  preprocessing metadata.
- Implemented the minimum behavior to make each red test pass.

## Verification

```text
PYTHONPATH=src python3 -m pytest tests/test_sampling.py tests/test_reproducibility.py tests/test_train_cli.py -v
7 passed in 4.23s
```

- `git diff --check` completed without output.
- `ruff` could not be run because it is not installed in the environment
  (`No module named ruff`).

## Scope

- No diagnostics or notebook changes were made.
- The commit contains only Task 2 implementation, tests, dependency metadata,
  and this report.

## Review Fix: Complete Run Bindings

- Added a `configuration` manifest section containing every resolved training CLI
  value, including absolute train/validation CSV paths, artifact directory, and
  resolved device.
- Added `datasets` metadata with the absolute CSV paths and train/validation
  dataset sizes after dataset construction and before model optimization.
- Added `environment` metadata with the resolved device and installed NumPy,
  Pillow, Torch, and Torchvision package versions.
- The same manifest continues to be written before epoch one and embedded in
  `best.pt`; train-only balancing and evaluation metadata rejection are unchanged.

### Review-Fix Verification

```text
PYTHONPATH=src python3 -m pytest tests/test_sampling.py tests/test_reproducibility.py tests/test_train_cli.py -v
7 passed in 4.00s
```

- The CLI integration test now asserts all configuration, dataset, device, and
  package-version bindings in both the manifest and checkpoint metadata.

## Re-Review Fix: Exact Package Versions

- Strengthened the CLI integration test to compare manifest package-version
  values against `importlib.metadata.version` for `numpy`, `Pillow`, `torch`,
  and `torchvision`, using the same distribution names as the manifest writer.
- This prevents a stale, substituted, or otherwise incorrect package version
  from satisfying the manifest contract merely by being a string.
