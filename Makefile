.PHONY: install demo validate train evaluate test lint reproduce clean

install:
	python -m pip install -e ".[dev]"

demo:
	microscan make-demo --output data/demo --force

validate:
	microscan validate-data --config configs/demo.yaml

train:
	microscan train --config configs/demo.yaml

evaluate:
	microscan evaluate --config configs/demo.yaml --checkpoint artifacts/demo/best_model.pt

test:
	pytest

lint:
	ruff check .

reproduce:
	bash scripts/reproduce_demo.sh

clean:
	python scripts/clean_generated.py

