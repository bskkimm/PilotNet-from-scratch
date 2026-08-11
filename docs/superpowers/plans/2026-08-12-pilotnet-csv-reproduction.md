# PilotNet CSV Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, paper-facing offline PilotNet workflow for generic CSV driving logs.

**Architecture:** Put shared image processing and expanded CSV samples in `pilotnet.data`; keep train/eval orchestration in root entry points. Add a sampler, deterministic run utility, and diagnostics as focused modules. The notebook duplicates the implementation cell-by-cell without importing the package.

**Tech Stack:** Python 3.10+, PyTorch, TorchVision, Pillow, NumPy, pytest.

## Global Constraints

- Preserve the existing PilotNet five-convolution and four-linear-layer architecture.
- Accept required `image_path`/`steering` CSV columns; accept optional blank `left_image_path`/`right_image_path` columns.
- Crop source images using validated normalized vertical fractions before RGB-to-YUV conversion and `66x200` resize.
- Keep center-only CSV logs backward compatible.
- Apply balancing only to train loaders; keep validation and diagnostics sequential.
- Store data and generated artifacts only under ignored root paths.
- Keep the notebook independent of `pilotnet` imports.
- Do not add simulator or closed-loop evaluation code in this plan.

---

### Task 1: Shared YUV Preprocessing And Expanded Driving Logs

**Files:**
- Create: `src/pilotnet/data/preprocessing.py`
- Modify: `src/pilotnet/data/dataset.py`
- Modify: `src/pilotnet/data/__init__.py`
- Modify: `tests/test_dataset.py`
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `PreprocessConfig(crop_top_fraction: float = 0.0, crop_bottom_fraction: float = 0.0, height: int = 66, width: int = 200)` and `preprocess_image(image: PIL.Image.Image, config: PreprocessConfig) -> torch.Tensor`.
- Produces: `DrivingDataset(csv_path, *, augment=False, preprocess=PreprocessConfig(), side_camera_correction: float | None = None)` with `targets: list[float]`.
- Consumes: the existing `(image, steering)` dataset contract used by `train_epoch` and `evaluate`.

- [ ] **Step 1: Write failing preprocessing tests**

```python
def test_preprocess_crops_converts_to_yuv_and_resizes() -> None:
    image = Image.new("RGB", (20, 10), color=(255, 0, 0))
    config = PreprocessConfig(crop_top_fraction=0.2, crop_bottom_fraction=0.2)
    result = preprocess_image(image, config)
    assert result.shape == (3, 66, 200)
    assert result.dtype == torch.float32
    assert result.min() >= 0 and result.max() <= 1

@pytest.mark.parametrize("config", [
    PreprocessConfig(crop_top_fraction=-0.1),
    PreprocessConfig(crop_bottom_fraction=1.0),
    PreprocessConfig(crop_top_fraction=0.5, crop_bottom_fraction=0.5),
])
def test_preprocess_rejects_invalid_crop_fractions(config) -> None:
    with pytest.raises(ValueError):
        preprocess_image(Image.new("RGB", (20, 10)), config)
```

- [ ] **Step 2: Run the preprocessing tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_preprocessing.py -v`

Expected: FAIL because `pilotnet.data.preprocessing` does not exist.

- [ ] **Step 3: Implement preprocessing configuration and transform**

```python
@dataclass(frozen=True)
class PreprocessConfig:
    crop_top_fraction: float = 0.0
    crop_bottom_fraction: float = 0.0
    height: int = 66
    width: int = 200

    def __post_init__(self) -> None:
        if not 0 <= self.crop_top_fraction < 1 or not 0 <= self.crop_bottom_fraction < 1:
            raise ValueError("Crop fractions must be in [0, 1).")
        if self.crop_top_fraction + self.crop_bottom_fraction >= 1:
            raise ValueError("Crop fractions must leave at least one image row.")

def preprocess_image(image: Image.Image, config: PreprocessConfig) -> torch.Tensor:
    top = round(image.height * config.crop_top_fraction)
    bottom = image.height - round(image.height * config.crop_bottom_fraction)
    cropped = image.convert("RGB").crop((0, top, image.width, bottom)).convert("YCbCr")
    resized = transforms.resize(cropped, (config.height, config.width), InterpolationMode.BILINEAR, antialias=True)
    return transforms.to_tensor(resized)
```

- [ ] **Step 4: Add failing dataset expansion tests**

```python
def test_dataset_expands_available_side_cameras(tmp_path) -> None:
    for name in ("center.jpg", "left.jpg", "right.jpg"):
        Image.new("RGB", (200, 66)).save(tmp_path / name)
    (tmp_path / "log.csv").write_text(
        "image_path,steering,left_image_path,right_image_path\ncenter.jpg,0.1,left.jpg,right.jpg\n"
    )
    dataset = DrivingDataset(tmp_path / "log.csv", side_camera_correction=0.2)
    assert dataset.targets == [0.1, 0.3, -0.1]

def test_dataset_requires_correction_for_available_side_camera(tmp_path) -> None:
    Image.new("RGB", (200, 66)).save(tmp_path / "center.jpg")
    Image.new("RGB", (200, 66)).save(tmp_path / "left.jpg")
    (tmp_path / "log.csv").write_text(
        "image_path,steering,left_image_path\ncenter.jpg,0.1,left.jpg\n"
    )
    with pytest.raises(ValueError, match="side_camera_correction"):
        DrivingDataset(tmp_path / "log.csv")
```

- [ ] **Step 5: Implement sample expansion and shared transform**

Store each sample as `(path, steering)`. Parse `left_image_path` and `right_image_path` only when headers exist and the cell is nonblank. Raise `FileNotFoundError` from initialization for each nonexistent expanded path. Use `preprocess_image` in `__getitem__`; on reflection, horizontally flip the already-YUV PIL source before transform and negate the associated steering target.

- [ ] **Step 6: Run focused data tests**

Run: `PYTHONPATH=src python3 -m pytest tests/test_preprocessing.py tests/test_dataset.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the data contract**

```bash
git add src/pilotnet/data tests/test_preprocessing.py tests/test_dataset.py
git commit -m "feat: add YUV preprocessing and recovery data"
```

### Task 2: Balanced Sampling And Deterministic Run Metadata

**Files:**
- Create: `src/pilotnet/data/sampling.py`
- Create: `src/pilotnet/utils/__init__.py`
- Create: `src/pilotnet/utils/reproducibility.py`
- Modify: `src/pilotnet/data/__init__.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `train.py`
- Modify: `eval.py`
- Create: `tests/test_sampling.py`
- Create: `tests/test_reproducibility.py`
- Create: `tests/test_train_cli.py`

**Interfaces:**
- Produces: `build_balanced_sampler(targets: Sequence[float], bins: int, generator: torch.Generator | None = None) -> WeightedRandomSampler`.
- Produces: `seed_everything(seed: int, deterministic: bool) -> dict[str, object]` and `build_run_manifest(...) -> dict[str, object]`.
- Consumes: `DrivingDataset.targets`, `PreprocessConfig`, and existing `train_epoch`/`evaluate` functions.

- [ ] **Step 1: Write failing sampler and seed tests**

```python
def test_balanced_sampler_weights_rare_bins_more_heavily() -> None:
    sampler = build_balanced_sampler([0.0, 0.0, 0.0, 1.0], bins=2)
    assert sampler.weights[-1] > sampler.weights[0]

def test_seed_everything_repeats_torch_values() -> None:
    seed_everything(7, deterministic=True)
    first = torch.rand(3)
    seed_everything(7, deterministic=True)
    assert torch.equal(first, torch.rand(3))
```

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_sampling.py tests/test_reproducibility.py -v`

Expected: FAIL because sampling and reproducibility modules do not exist.

- [ ] **Step 3: Implement balanced sampler and reproducibility utilities**

```python
def build_balanced_sampler(targets: Sequence[float], bins: int, generator=None) -> WeightedRandomSampler:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    indices = torch.bucketize(torch.tensor(targets), torch.linspace(min(targets), max(targets), bins + 1)[1:-1])
    counts = torch.bincount(indices, minlength=bins).clamp_min(1)
    weights = counts[indices].reciprocal().to(torch.double)
    return WeightedRandomSampler(weights, len(targets), replacement=True, generator=generator)

def seed_everything(seed: int, deterministic: bool) -> dict[str, object]:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    return {"seed": seed, "deterministic": deterministic}
```

Handle the all-identical-target case as one uniform bin rather than passing duplicate boundaries to `bucketize`.

- [ ] **Step 4: Add manifest and CLI integration tests**

```python
def test_training_writes_preprocessing_to_manifest(tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (200, 66)).save(image)
    for name in ("train.csv", "val.csv"):
        (tmp_path / name).write_text("image_path,steering\nframe.jpg,0.0\n")
    output_dir = tmp_path / "run"
    subprocess.run([
        sys.executable, "train.py", "--train-csv", str(tmp_path / "train.csv"),
        "--val-csv", str(tmp_path / "val.csv"), "--output-dir", str(output_dir),
        "--epochs", "1", "--workers", "0", "--crop-top-fraction", "0.1", "--balance-bins", "5",
    ], check=True)
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["preprocessing"]["crop_top_fraction"] == 0.1
    assert manifest["sampling"]["balance_bins"] == 5
```

- [ ] **Step 5: Integrate train/eval configuration**

Add NumPy to both dependency declarations. Add CLI options for crop fractions, side-camera correction, balancing bins, deterministic mode, seed, artifact directory, and worker count. Construct the train loader with `sampler=...` and no `shuffle`; construct the validation loader with `shuffle=False` and no sampler. Write `run_manifest.json` before epoch one and include it in `best.pt`. Make `eval.py` load preprocessing from checkpoint metadata; fail if metadata is absent.

- [ ] **Step 6: Run sampling, reproducibility, and CLI tests**

Run: `PYTHONPATH=src python3 -m pytest tests/test_sampling.py tests/test_reproducibility.py tests/test_train_cli.py -v`

Expected: PASS.

- [ ] **Step 7: Commit reproducible training**

```bash
git add src/pilotnet/data src/pilotnet/utils pyproject.toml requirements.txt train.py eval.py tests
git commit -m "feat: add balanced reproducible training"
```

### Task 3: Prediction And Saliency Diagnostics

**Files:**
- Create: `src/pilotnet/diagnostics/__init__.py`
- Create: `src/pilotnet/diagnostics/visualization.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `eval.py`
- Create: `tests/test_diagnostics.py`

**Interfaces:**
- Produces: `save_prediction_panel(model, dataset, indices, output_path, device) -> Path`.
- Produces: `save_saliency_overlay(model, image, output_path, device) -> Path`.
- Consumes: YUV image tensors from `DrivingDataset`; diagnostic output path supplied by CLI.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_prediction_panel_is_written(tmp_path, dataset, model) -> None:
    path = save_prediction_panel(model, dataset, [0], tmp_path / "panel.png", torch.device("cpu"))
    assert path.exists()
    assert Image.open(path).width > 0

def test_saliency_overlay_is_written(tmp_path, model) -> None:
    path = save_saliency_overlay(model, torch.rand(3, 66, 200), tmp_path / "saliency.png", torch.device("cpu"))
    assert Image.open(path).size == (200, 66)
```

- [ ] **Step 2: Run diagnostic tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_diagnostics.py -v`

Expected: FAIL because `pilotnet.diagnostics` does not exist.

- [ ] **Step 3: Implement visual diagnostics**

Add Matplotlib to both dependency declarations. Use Matplotlib with the noninteractive `Agg` backend. Convert YUV tensors to RGB before display. In saliency, clone the single image with `requires_grad_(True)`, backpropagate its scalar steering output, calculate `gradient.abs().amax(dim=1)`, normalize only when the maximum is positive, and alpha-blend over the RGB image. Reject empty index lists and images not shaped `(3, 66, 200)` with `ValueError`.

- [ ] **Step 4: Add evaluation CLI coverage and integration**

Add `--prediction-indices`, `--prediction-output`, and `--saliency-output` options. Parse comma-separated nonnegative indices, require that all indices are below `len(dataset)`, and create only requested artifact directories. Test CLI output paths with a one-sample fixture.

- [ ] **Step 5: Run diagnostic tests**

Run: `PYTHONPATH=src python3 -m pytest tests/test_diagnostics.py -v`

Expected: PASS.

- [ ] **Step 6: Commit diagnostics**

```bash
git add src/pilotnet/diagnostics pyproject.toml requirements.txt eval.py tests/test_diagnostics.py tests/test_train_cli.py
git commit -m "feat: add prediction and saliency diagnostics"
```

### Task 4: Standalone Notebook Parity

**Files:**
- Modify: `notebooks/pilotnet_walkthrough.ipynb`
- Modify: `tests/test_notebook.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the exact public CSV contract and configuration defaults implemented in Tasks 1-3.
- Produces: a notebook that independently defines the preprocessing, data expansion, sampler, deterministic seed, manifest, prediction panel, and saliency functions.

- [ ] **Step 1: Extend the failing structural notebook test**

```python
def test_notebook_defines_reproduction_workflow() -> None:
    code = notebook_code()
    for name in ["PreprocessConfig", "preprocess_image", "DrivingDataset", "build_balanced_sampler", "seed_everything", "save_saliency_overlay"]:
        assert f"def {name}" in code or f"class {name}" in code
    assert "from pilotnet" not in code
```

- [ ] **Step 2: Run the notebook structural test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_notebook.py -v`

Expected: FAIL because the standalone notebook lacks the new functions.

- [ ] **Step 3: Add independent notebook cells**

Add separate explanatory and code-cell pairs for normalized crop/YUV conversion, center/side camera expansion, inverse-frequency sampling, deterministic seed and manifest construction, prediction panels, and saliency. Each code cell must define its own function/class and use only standard-library, PyTorch, TorchVision, Pillow, NumPy, and Matplotlib imports.

- [ ] **Step 4: Update concise README usage**

Document optional CSV side-camera columns, YUV/crop configuration, sampling option, manifest location, diagnostics commands, and the fact that the notebook is self-contained.

- [ ] **Step 5: Run notebook and documentation checks**

Run: `python3 -m json.tool notebooks/pilotnet_walkthrough.ipynb > /dev/null && PYTHONPATH=src python3 -m pytest tests/test_notebook.py -v`

Expected: PASS.

- [ ] **Step 6: Commit notebook parity**

```bash
git add notebooks/pilotnet_walkthrough.ipynb tests/test_notebook.py README.md
git commit -m "feat: extend standalone reproduction notebook"
```

### Task 5: Full Regression Verification

**Files:**
- Modify: `README.md` only if a command or data contract is inaccurate after implementation.
- Test: `tests/`

**Interfaces:**
- Consumes: all completed modules and entry points.
- Produces: verified installation, test suite, CLI help, valid notebook JSON, and a clean worktree.

- [ ] **Step 1: Install the declared development dependencies in an isolated environment**

Run: `python3 -m pip install -e '.[dev]'`

Expected: package and development tools install without modifying tracked files.

- [ ] **Step 2: Run all tests**

Run: `python3 -m pytest -v`

Expected: PASS, including preprocessing, recovery expansion, balancing, reproducibility, diagnostics, and notebook tests.

- [ ] **Step 3: Run static and entry-point checks**

Run: `python3 -m ruff check src tests train.py eval.py && python3 train.py --help && python3 eval.py --help && python3 -m json.tool notebooks/pilotnet_walkthrough.ipynb > /dev/null`

Expected: all commands exit zero.

- [ ] **Step 4: Inspect the final worktree and commit only necessary documentation correction**

Run: `git status --short && git diff --check`

Expected: empty status; if README correction is needed, commit it as `docs: finalize reproduction workflow guide` and rerun this step.
