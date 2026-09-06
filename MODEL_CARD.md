# Model card: MicroScan TinyCNN baseline

## Status

**Software baseline only — not scientifically validated and not suitable for clinical,
environmental, regulatory, or operational decisions.**

The repository did not contain an experimental protocol, dataset, label definitions,
or previously validated model. This card therefore documents a configurable baseline,
not a claim about microplastic detection performance.

## Intended use

- Verify that a microscopy image-classification dataset can be loaded, trained,
  evaluated, and used for single-image inference.
- Establish a reproducible reference implementation for later comparison.
- Support independent validation after real data and a study protocol are supplied.

## Model

`TinyCNN` has three convolutional blocks, global average pooling, and a linear
classification head. Input channels, image size, model width, and classes are
configuration- or dataset-driven. Images are resized and scaled to `[0, 1]`; no
domain-specific preprocessing is claimed.

## Data

No research dataset ships with the repository. The generated demo contains simple
synthetic geometric patterns with arbitrary labels (`class_a`, `class_b`, `class_c`).
It exists only for software tests. Its accuracy must never be reported as evidence
of microplastic performance.

## Evaluation

The pipeline reports test loss, accuracy, macro F1, per-class precision/recall/F1,
support, and a confusion matrix. The held-out test split is evaluated separately.
An independent study should pre-register its endpoints and confidence intervals;
these are not chosen here because the experimental design is unknown.

## Limitations and risks

- No external validation, calibration, robustness, subgroup, or acquisition-shift study.
- Resizing can remove diagnostically relevant detail.
- Folder labels can encode leakage or acquisition artifacts.
- Exact-byte duplicate checks do not detect near-duplicates or correlated crops.
- Scores from `predict` are not calibrated probabilities.
- Small CNN capacity is a baseline choice, not an architecture recommendation.

See `docs/VALIDATION.md` before using real data.

