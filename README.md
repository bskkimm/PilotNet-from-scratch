# PilotNet from Scratch

A clean reproduction study of NVIDIA's 2016 PilotNet, *End to End Learning for Self-Driving Cars*. The model maps one RGB front-camera image to a continuous steering prediction.

## Contract

- Reimplement the camera-to-steering convolutional policy and behavioral-cloning workflow.
- Evaluate only on datasets and metrics that can be documented and reproduced publicly.
- Do not claim reproduction of NVIDIA's private data collection or reported driving performance.
- Keep data, checkpoints, and generated experiment artifacts out of Git.

## Reference

M. Bojarski et al., [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316), 2016.

## Status

The initial model, CSV driving-log workflow, and validation entry point are implemented.
Use `notebooks/pilotnet_walkthrough.ipynb` for the architecture and training walkthrough.

## Install

```bash
python3 -m pip install -e '.[dev]'
```

Install optional nuScenes extraction and MLflow support when needed:

```bash
python3 -m pip install -e '.[nuscenes,mlflow]'
```

## Data

Training and validation each use a CSV with `image_path` and `steering` columns. Image paths are relative to the CSV file. Images are resized to `200x66`; training applies horizontal flips and negates the matching steering target.

```csv
image_path,steering
images/frame_0001.jpg,-0.12
```

Keep related frames in the same split to avoid route or temporal leakage.

### Udacity Behavioral Cloning

NVIDIA did not release the original PilotNet training data. The public Udacity
Behavioral Cloning Kaggle mirror is a documented substitute, not the original dataset:

```text
https://www.kaggle.com/datasets/hsuchialun/carndbehavioralcloningp3
```

Install the optional downloader, then download the mirror to a directory you choose.
KaggleHub may require signing in to Kaggle and accepting the mirror's license.

```bash
python3 -m pip install -e '.[udacity]'
python3 scripts/download_udacity_pilotnet.py --output-dir ~/dataset/udacity_carnd
```

The helper preserves `driving_log.csv` and `IMG/` under that directory, then writes
prepared CSVs to `~/dataset/udacity_carnd/pilotnet/`. To prepare an already-downloaded copy:

```bash
python3 scripts/prepare_udacity_pilotnet.py \
  --driving-log /path/to/udacity/driving_log.csv \
  --output-dir data/udacity
```

The converter uses a contiguous tail validation split to avoid random temporal
leakage. Its output keeps left/right camera paths, so train with a documented
recovery correction:

```bash
python3 train.py --train-csv data/udacity/train.csv --val-csv data/udacity/val.csv \
  --side-camera-correction 0.2
```

## Run

```bash
python3 train.py --train-csv data/train.csv --val-csv data/val.csv
python3 eval.py --checkpoint artifacts/train/best.pt --csv data/val.csv
```

Track a CLI run with MLflow:

```bash
python3 train.py --train-csv data/train.csv --val-csv data/val.csv \
  --mlflow-tracking-uri sqlite:///artifacts/mlflow.db \
  --mlflow-experiment PilotNet --mlflow-run-name baseline
```

## Layout

```text
src/pilotnet/models/   # PilotNet architecture
src/pilotnet/data/     # CSV driving-log dataset
src/pilotnet/engine/   # training and validation routines
notebooks/             # guided learner walkthrough
```
