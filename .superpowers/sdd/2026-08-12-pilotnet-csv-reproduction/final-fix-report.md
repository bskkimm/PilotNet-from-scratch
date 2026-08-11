# PilotNet Final Fix Report

## Status

All final review findings for the delivered Task 1/2 scope are addressed.

- `DrivingDataset` rejects non-finite steering values with a field-specific
  `ValueError`, validates finite nonnegative side-camera correction values when
  side cameras are populated, and verifies plus decodes every image during
  construction.
- Training passes the configured side-camera correction to validation while the
  validation loader remains sequential with `shuffle=False` and no sampler.
- `PreprocessConfig` rejects nonpositive output dimensions.
- The run manifest records `environment["python"]` from `sys.version`.
- PilotNet public input documentation identifies YUV/YCbCr tensors.

## Tests

```text
PYTHONPATH=src python3 -m pytest -v
33 passed in 4.30s
```

Focused regression coverage includes finite steering values, non-image and
truncated-image paths, negative side-camera corrections, positive output
dimensions, populated validation side cameras with a sequential loader, and
the exact Python version recorded in the manifest.

## Concerns

`PYTHONPATH=src python3 -m ruff check .` could not run because `ruff` is not
installed in the active Python environment. The full pytest suite completed
successfully.
