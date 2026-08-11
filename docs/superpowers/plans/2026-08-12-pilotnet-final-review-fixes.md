# PilotNet Final Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all final review findings for CSV validation, preprocessing configuration, run metadata, and public input documentation.

**Architecture:** Keep validation at data-construction boundaries so invalid CSV values and images fail before loader creation. Keep training and validation dataset construction symmetric for recovery cameras while preserving the existing balanced-only training sampler. Extend the existing manifest environment object with the active Python version.

**Tech Stack:** Python, PyTorch, Pillow, pytest.

## Global Constraints

- Implement directly on `main`.
- Deliver all changes and the required report in one final fix commit.
- Preserve sequential validation loading with `shuffle=False` and no sampler.
- Record the interpreter as `environment["python"] = sys.version`.

---

### Task 1: Validate Dataset and Preprocessing Inputs

**Files:**
- Modify: `src/pilotnet/data/dataset.py:5-59`
- Modify: `src/pilotnet/data/preprocessing.py:20-24`
- Test: `tests/test_dataset.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `DrivingDataset(csv_path, *, augment=False, preprocess=PreprocessConfig(), side_camera_correction=None)` that raises field-specific `ValueError` for non-finite steering and invalid side-camera correction and verifies all image files during construction.
- Produces: `PreprocessConfig(...)` that requires positive `height` and `width`.

- [ ] **Step 1: Write failing dataset and preprocessing tests**

```python
with pytest.raises(ValueError, match="steering"):
    DrivingDataset(csv_path)

with pytest.raises(ValueError, match="side_camera_correction"):
    DrivingDataset(csv_path, side_camera_correction=-0.1)

with pytest.raises(ValueError, match="height"):
    PreprocessConfig(height=0)
```

- [ ] **Step 2: Run focused tests to verify failures**

Run: `PYTHONPATH=src python3 -m pytest tests/test_dataset.py tests/test_preprocessing.py -v`
Expected: FAIL because invalid finite values, corrupt images, and nonpositive dimensions are accepted.

- [ ] **Step 3: Add boundary validation**

```python
if not math.isfinite(steering):
    raise ValueError(f"Invalid steering value in row {row_number}, field 'steering'.")

with Image.open(image_path) as image_file:
    image_file.verify()
```

- [ ] **Step 4: Run focused tests to verify success**

Run: `PYTHONPATH=src python3 -m pytest tests/test_dataset.py tests/test_preprocessing.py -v`
Expected: PASS.

### Task 2: Propagate Validation Recovery Configuration and Metadata

**Files:**
- Modify: `train.py:5-105`
- Modify: `src/pilotnet/models/pilotnet.py:10-15`
- Test: `tests/test_train_cli.py`
- Test: `tests/test_reproducibility.py`

**Interfaces:**
- Consumes: `DrivingDataset(..., side_camera_correction=args.side_camera_correction)`.
- Produces: `run_manifest["environment"]["python"]` equal to `sys.version`.

- [ ] **Step 1: Write failing manifest and side-camera validation tests**

```python
assert manifest["environment"]["python"] == sys.version
assert manifest["datasets"]["val_size"] == 3
```

- [ ] **Step 2: Run focused tests to verify failures**

Run: `PYTHONPATH=src python3 -m pytest tests/test_train_cli.py tests/test_reproducibility.py -v`
Expected: FAIL because the manifest lacks Python and validation lacks configured correction.

- [ ] **Step 3: Implement CLI and documentation corrections**

```python
val_dataset = DrivingDataset(
    args.val_csv,
    preprocess=preprocessing,
    side_camera_correction=args.side_camera_correction,
)
```

- [ ] **Step 4: Run focused tests to verify success**

Run: `PYTHONPATH=src python3 -m pytest tests/test_train_cli.py tests/test_reproducibility.py -v`
Expected: PASS.

### Task 3: Verify and Deliver

**Files:**
- Create: `.superpowers/sdd/2026-08-12-pilotnet-csv-reproduction/final-fix-report.md`

- [ ] **Step 1: Run the complete test suite**

Run: `PYTHONPATH=src python3 -m pytest -v`
Expected: PASS with all tests collected.

- [ ] **Step 2: Write the final fix report**

```markdown
# Final Fix Report

## Status

All requested final review findings are addressed.
```

- [ ] **Step 3: Commit the complete fix wave**

```bash
git add src/pilotnet/data/dataset.py src/pilotnet/data/preprocessing.py src/pilotnet/models/pilotnet.py train.py tests .superpowers/sdd docs/superpowers/plans
```

## Self-Review

- Spec coverage: Task 1 covers finite steering, image verification, nonnegative correction, and positive dimensions. Task 2 covers validation recovery support, sequential validation preservation, Python metadata, and YUV/YCbCr documentation. Task 3 covers full verification, report, and one commit.
- Placeholder scan: no incomplete implementation requirements remain.
- Type consistency: the existing `DrivingDataset` and `PreprocessConfig` interfaces remain unchanged.
