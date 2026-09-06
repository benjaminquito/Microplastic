# Microplastic / MicroScan AI

A reproducible, testable **image-classification baseline** for microscopy research.
The project is deliberately conservative: no original dataset, label definitions,
experimental protocol, model architecture, or reported results were available when
it was created, so none are invented here.

> **Current status:** the software pipeline is runnable and testable. It is not a
> scientifically validated microplastic model. Synthetic demo metrics prove only
> that the code executes end to end.

## What is included

- Deterministic synthetic data generator for software verification
- Strict `train` / `val` / `test` image-folder contract
- Dataset checks for readability, class consistency, empty classes, and exact-copy leakage
- Small configurable PyTorch CNN baseline
- Reproducible training with seeded randomness and recorded run metadata
- Held-out evaluation: loss, accuracy, macro F1, per-class metrics, confusion matrix
- Versioned checkpoints with preprocessing and class metadata
- Single-image JSON inference
- Unit tests, end-to-end smoke test, linting, and GitHub Actions CI
- Model card and independent-validation protocol template

## Repository layout

```text
configs/                  Demo and real-data configuration templates
data/                     Dataset contract; generated/private data are ignored
docs/VALIDATION.md        Protocol fields required for credible validation
scripts/reproduce_demo.sh One-command synthetic reproduction
src/microscan_ai/         Data, model, training, evaluation, and inference code
tests/                    Unit and end-to-end tests
MODEL_CARD.md             Intended use, limitations, and evidence status
```

## Quick start

Python 3.10–3.12 is supported. CPU execution is the default so the demo is portable.

```bash
git clone https://github.com/benjaminquito/Microplastic.git
cd Microplastic
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
bash scripts/reproduce_demo.sh
```

Generated evidence is written to `artifacts/demo/`:

- `data_validation.json`: dataset structure/integrity report
- `run_manifest.json`: seed, effective config, class counts, and environment
- `history.json`: per-epoch training/validation metrics
- `best_model.pt`: checkpoint selected only by validation loss
- `test_metrics.json`: final held-out test results

Data and artifacts are git-ignored. Re-running the command regenerates them.

## Use your dataset

1. Read `data/README.md` and structure data as:

   ```text
   your_dataset/{train,val,test}/{your_class_name}/*.{png,jpg,tif,...}
   ```

2. Copy `configs/real_data.template.yaml` to `configs/local-real.yaml` and set the
   dataset path and justified hyperparameters. Local configs are ignored by git.
3. Validate before training:

   ```bash
   microscan validate-data --config configs/local-real.yaml \
     --output artifacts/real-data/data_validation.json
   ```

4. Train, then evaluate the frozen checkpoint once on the held-out test split:

   ```bash
   microscan train --config configs/local-real.yaml
   microscan evaluate --config configs/local-real.yaml \
     --checkpoint artifacts/real-data/best_model.pt
   ```

5. Run inference:

   ```bash
   microscan predict --checkpoint artifacts/real-data/best_model.pt \
     --image /path/to/image.tif --device cpu
   ```

Class names are discovered from folder names; no scientific taxonomy is hardcoded.

## Reproducibility guarantees and boundaries

- The seed is applied to Python, NumPy, and PyTorch, and deterministic PyTorch
  algorithms are requested.
- Demo execution is pinned to CPU with zero data-loader workers.
- The checkpoint stores architecture, width, channels, image size, classes, seed,
  selected epoch, and validation loss.
- Each run records package/platform versions and the effective configuration.
- Direct dependencies are exactly pinned. For archival replication, also preserve
  the environment's full dependency export or a container image digest.

Bitwise equality is not promised across different hardware, operating systems, or
PyTorch builds. Scientific replication also requires the exact dataset version,
split manifest, acquisition protocol, and analysis plan.

## Testing

```bash
pytest
ruff check .
```

CI runs both commands on every push and pull request.

## What is still required from the study owner

No legitimate research metric can be produced until these are supplied:

- Original microscopy images and usage rights
- Prediction target and operational class definitions
- Sample/specimen identifiers needed for leakage-safe grouped splits
- Acquisition and preprocessing protocol
- Label provenance and annotation quality-control records
- Pre-specified endpoints, acceptance criteria, and validation population
- Independent external dataset, if generalization is claimed

Complete `docs/VALIDATION.md` and update `MODEL_CARD.md` with this information.
Do not commit restricted data or secrets; `.env` and dataset contents are ignored.

## Scope

This repository currently solves image classification. Switching to object detection,
semantic/instance segmentation, particle counting, or spectroscopy requires labels
and an experimental specification for that task. Such a change should be made only
after those inputs are available.

