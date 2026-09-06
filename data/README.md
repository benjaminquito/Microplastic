# Dataset contract

No experimental data is included in this repository. Do not commit confidential,
restricted, or identifying data.

Place images in explicit, mutually exclusive splits:

```text
dataset/
├── train/
│   ├── class_name_1/
│   └── class_name_2/
├── val/
│   ├── class_name_1/
│   └── class_name_2/
└── test/
    ├── class_name_1/
    └── class_name_2/
```

Requirements:

- Every split must contain the same class-directory names.
- Each class must contain at least one readable image in every split.
- The same image bytes must not appear in more than one split.
- Splits should be made at the independent experimental-unit level (for example,
  specimen or acquisition session), not by randomly separating correlated crops.
- Document sampling, imaging, preprocessing, label definitions, annotators,
  disagreements, exclusions, and the split procedure before reporting results.

Run `microscan validate-data --config <config>` before training. It detects unreadable
files, inconsistent classes, and exact duplicate leakage between splits.

`microscan make-demo` creates a deterministic synthetic dataset named `class_a`,
`class_b`, and `class_c`. Those shapes test code paths only; they are not
microplastic observations and cannot validate scientific performance.

