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