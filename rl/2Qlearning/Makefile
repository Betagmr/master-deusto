PY = python3
PI = pip3
VENV = .venv
BIN=$(VENV)/bin

ifeq ($(OS), Windows_NT)
    BIN=$(VENV)/Scripts
    PY=python
endif

PYTHON = $(BIN)/$(PY)
PIP = $(BIN)/$(PI)

install:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .   

run:
	$(PYTHON) -m src.main

optuna:
	$(PYTHON) -m src.optimizer

optuna-frozen:
	$(PYTHON) -m optuna-dashboard sqlite:///frozen-study.db

optuna-taxi:
	$(PYTHON) -m optuna-dashboard sqlite:///taxi-study.db