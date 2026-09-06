# Independent validation protocol template

Complete this document with the study owners before treating any run as research
evidence. Blank items are intentionally not inferred.

## 1. Research question

- Intended prediction target: **TBD**
- Unit of analysis: **TBD**
- Intended population and setting: **TBD**
- Intended decision or use: **TBD**
- Primary endpoint and acceptance threshold: **TBD**

## 2. Dataset provenance

Record dataset owner, collection dates/sites, sampling method, microscope/camera,
magnification, resolution, illumination, preparation/staining, inclusion/exclusion
criteria, licenses/consent, and a stable dataset version or checksum.

## 3. Labels

Define every class operationally. Record annotator qualifications, blinding,
adjudication, agreement measurement, uncertain-label policy, and quality control.

## 4. Split integrity

Split by the highest independent unit needed to prevent leakage (for example,
specimen, bottle, site, acquisition session, or study). Keep the test set sealed
until model and decision rules are frozen. Run the exact-duplicate validator and
add a domain-appropriate near-duplicate/group audit.

## 5. Analysis plan

Pre-specify the primary metric, uncertainty interval, class-imbalance handling,
missing/corrupt data policy, hyperparameter selection, number of runs/seeds, and
all subgroup and sensitivity analyses. Report every attempted model or distinguish
exploratory analyses clearly.

## 6. External validation

Use a separately collected dataset from the intended deployment conditions. Do not
tune on it. Compare performance and calibration with the internal test set and
investigate acquisition, temporal, geographic, and operator shifts.

## 7. Reproduction record

Archive, subject to permissions:

- Git commit SHA and unchanged configuration
- Dataset version/checksums and split manifest
- `artifacts/<run>/run_manifest.json`
- Model checkpoint and `test_metrics.json`
- Hardware/OS information and complete dependency lock or container digest
- Deviations from the pre-specified protocol

## What the repository validates today

The automated tests validate configuration checks, image loading, duplicate-leakage
detection, model tensor shape/backpropagation, metric calculations, checkpointing,
and a one-epoch train/evaluate smoke run on synthetic data. They do not validate
scientific accuracy or fitness for real-world use.

