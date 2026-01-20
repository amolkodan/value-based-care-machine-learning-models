PYTHON ?= python
VENV = .venv

.PHONY: venv install lint test

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check src tests

test:
	$(VENV)/bin/pytest -q
