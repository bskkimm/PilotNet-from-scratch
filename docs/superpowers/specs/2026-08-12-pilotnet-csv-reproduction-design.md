# PilotNet Generic CSV Reproduction Design

## Goal

Make the current PilotNet implementation a reproducible offline behavioral-cloning study for public, generic CSV driving logs. The phase improves input fidelity, recovery supervision, experiment repeatability, and offline analysis without claiming NVIDIA dataset or closed-loop result reproduction.

## Scope

- Convert source RGB images to YUV after optional normalized vertical cropping, then resize to `66x200`.
- Support center-only logs and optional left/right recovery cameras.
- Balance training steering-angle bins without altering validation sampling.
- Persist a complete run manifest with deterministic seed state and resolved inputs.
- Produce prediction and gradient-saliency image artifacts.
- Teach the same operations in the standalone notebook.

## Non-Goals

- CARLA, Udacity simulator, or another closed-loop integration.
- Reproducing NVIDIA's private collection process or reported results.
- A mandatory public dataset adapter.
- Changing the five-convolution, four-linear-layer PilotNet architecture.

## Data Contract

The CSV requires `image_path` and `steering`. Paths resolve relative to the CSV file. `left_image_path` and `right_image_path` are optional and may be blank per row.

`PreprocessConfig` defines `crop_top_fraction`, `crop_bottom_fraction`, and target height/width. Fractions are each in `[0, 1)`, and their sum must be less than one. The default preserves the complete image. Crop bounds are calculated from the source image height before RGB-to-YUV conversion and resizing.

The training dataset emits one center sample for every row. If an available side-camera path is nonblank, it emits one additional sample. Left steering is `center_steering + correction`; right steering is `center_steering - correction`. The correction is a required nonnegative configuration value when either side camera column is used. Invalid images, nonnumeric steering values, invalid crop settings, and side-camera columns without a correction fail before training begins.

## Components

`pilotnet.data.preprocessing` owns validated preprocessing configuration and the shared PIL/Tensor transform. Its output is a YUV float tensor in `[0, 1]` with shape `(3, 66, 200)`.

`DrivingDataset` owns CSV parsing, relative path resolution, side-camera sample expansion, optional horizontal reflection, and target creation. Reflection negates steering after any side-camera correction.

`BalancedSteeringSampler` receives expanded training targets and a configured bin count. It assigns inverse-frequency weights by bin and uses replacement sampling for one epoch equal to the expanded dataset size. It is never applied to validation or diagnostics loaders.

`ExperimentConfig` is the single resolved configuration passed to entry points. It includes preprocessing, side-camera correction, sampler settings, seed, optimizer parameters, and loader settings. `seed_everything` seeds Python, NumPy, and PyTorch CPU/CUDA generators and enables deterministic PyTorch algorithms when requested.

The training entry point writes `run_manifest.json` before the first optimization step. The manifest contains the resolved configuration, Python/package versions, device, train and validation CSV paths, and dataset sizes. Best checkpoints embed the same configuration and validation metrics.

`pilotnet.diagnostics` writes two explicit artifacts: a prediction panel for chosen validation examples and a gradient-saliency overlay. Saliency is the absolute gradient of the scalar steering output with respect to the YUV input, reduced across channels and normalized per image. It is a diagnostic, not an explanation of causal driving behavior.

## Entry Points And Notebook

`train.py` accepts preprocessing, recovery, balancing, seed, and artifact options. `eval.py` reconstructs preprocessing from the checkpoint manifest unless explicitly overridden with matching values.

The standalone notebook rebuilds preprocessing, dataset expansion, sampler, run metadata, prediction visualization, and saliency in cells. It remains independent of imports from `pilotnet`.

## Error Handling

- Fail with a field-specific `ValueError` for invalid CSV headers, steering values, crop fractions, correction values, or bin counts.
- Fail with `FileNotFoundError` naming the unresolved image path before a training epoch starts.
- Refuse evaluation when a checkpoint lacks preprocessing metadata; silently guessing input preprocessing would invalidate the result.
- Create artifact directories when requested and never write artifacts into tracked source directories.

## Tests

- Verify YUV conversion, crop geometry, output shape, and malformed preprocessing configuration.
- Verify center-only compatibility, optional side-camera expansion, steering corrections, and reflection behavior.
- Verify balanced weights favor rare steering bins and validation remains sequential.
- Verify seeded CPU sampling/training is repeatable for a fixed fixture.
- Verify manifests and checkpoints contain resolved preprocessing and metrics.
- Verify prediction and saliency artifact dimensions and file creation.
- Extend the notebook structural test to reject `pilotnet` imports while compiling all independent code cells.

## Delivery Boundaries

Implementation commits are organized by preprocessing/data contract, balancing and reproducibility, diagnostics, notebook parity, and tests/documentation. Closed-loop evaluation begins only in a separate design and implementation cycle.
