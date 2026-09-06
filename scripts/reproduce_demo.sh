#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=20260906

microscan make-demo --output data/demo --seed 20260906 --force
microscan validate-data --config configs/demo.yaml --output artifacts/demo/data_validation.json
microscan train --config configs/demo.yaml
microscan evaluate --config configs/demo.yaml --checkpoint artifacts/demo/best_model.pt
pytest

echo "Demo reproduction complete. Results are in artifacts/demo/."

