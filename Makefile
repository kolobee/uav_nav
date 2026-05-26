# UAV Navigation Project — Makefile
# Usage: make <target>
# Requires Python 3.11 virtualenv at ./venv (or active conda env).

PYTHON      := python
PIP         := pip
PYTEST      := pytest
BLACK       := black
RUFF        := ruff
PRE_COMMIT  := pre-commit

# Config paths
CONFIG_DIR  := uav_nav/configs
DEFAULT_CFG := $(CONFIG_DIR)/default.yaml
PI5_CFG     := $(CONFIG_DIR)/pi5.yaml

# Output directories
LOG_DIR     := logs
RESULTS_DIR := results
WEIGHTS_DIR := weights
DATA_DIR    := data

.PHONY: help install install-dev lint format check test test-fast \
        train_yolo train_embedding build_tilm run_experiment \
        benchmark_pi5 export_onnx export_ncnn clean clean-all \
        pre-commit-install pre-commit-run

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo "UAV Navigation — available Makefile targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install           Install package in editable mode"
	@echo "    install-dev       Install with dev + viz extras"
	@echo "    pre-commit-install Install pre-commit hooks"
	@echo ""
	@echo "  Code quality:"
	@echo "    lint              Run ruff linter"
	@echo "    format            Run black formatter"
	@echo "    check             Run lint + format check (no changes)"
	@echo "    pre-commit-run    Run all pre-commit hooks on staged files"
	@echo ""
	@echo "  Tests:"
	@echo "    test              Run full pytest suite with coverage"
	@echo "    test-fast         Run tests excluding slow/dataset markers"
	@echo ""
	@echo "  Pipeline:"
	@echo "    train_yolo        Fine-tune YOLO segmentation on MidAir"
	@echo "    train_embedding   Train semantic embedding head"
	@echo "    build_tilm        Build Topological Invariant Landmark Map"
	@echo "    run_experiment    Run all dissertation experiments"
	@echo "    benchmark_pi5     Benchmark pipeline on Pi 5 hardware"
	@echo ""
	@echo "  Deployment:"
	@echo "    export_onnx       Export models to ONNX format"
	@echo "    export_ncnn       Convert ONNX models to NCNN format"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean             Remove build artifacts and __pycache__"
	@echo "    clean-all         Also remove logs, results, and weights"

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,viz]"

pre-commit-install:
	$(PRE_COMMIT) install

# ── Code Quality ─────────────────────────────────────────────────────────────
lint:
	$(RUFF) check uav_nav/ --select E,F,I --line-length 100

format:
	$(BLACK) uav_nav/ --line-length 100

check:
	$(BLACK) uav_nav/ --line-length 100 --check
	$(RUFF) check uav_nav/ --select E,F,I --line-length 100

pre-commit-run:
	$(PRE_COMMIT) run --all-files

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) uav_nav/tests/ \
		--cov=uav_nav \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v

test-fast:
	$(PYTEST) uav_nav/tests/ \
		-m "not slow and not requires_midair and not requires_gpu and not requires_airsim" \
		--tb=short \
		-v

# ── Pipeline Stages ───────────────────────────────────────────────────────────
train_yolo:
	@echo ">>> Step 1/2: Building dataset (if needed)..."
	$(PYTHON) uav_nav/scripts/01_build_dataset.py
	@echo ">>> Step 2/2: Training YOLO segmentation model..."
	$(PYTHON) uav_nav/scripts/02_train_yolo.py

train_embedding:
	@echo ">>> Training semantic embedding head..."
	$(PYTHON) uav_nav/scripts/03_train_embedding.py

build_tilm:
	@echo ">>> Building Topological Invariant Landmark Map (TILM)..."
	$(PYTHON) uav_nav/scripts/04_build_tilm.py

run_experiment:
	@echo ">>> Running all dissertation experiments..."
	$(PYTHON) uav_nav/scripts/08_run_all_experiments.py

benchmark_pi5:
	@echo ">>> Benchmarking pipeline on Raspberry Pi 5..."
	$(PYTHON) uav_nav/scripts/07_benchmark_pi5.py \
		--config-name pi5

# ── Deployment ────────────────────────────────────────────────────────────────
export_onnx:
	@echo ">>> Exporting models to ONNX..."
	$(PYTHON) -c "\
from pathlib import Path; \
from uav_nav.deployment.export_onnx import export_to_onnx, ONNXExportConfig; \
print('export_onnx: implement in deployment/export_onnx.py')"

export_ncnn:
	@echo ">>> Converting ONNX models to NCNN..."
	$(PYTHON) -c "\
from pathlib import Path; \
from uav_nav.deployment.export_ncnn import export_to_ncnn, NCNNExportConfig; \
print('export_ncnn: implement in deployment/export_ncnn.py')"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "Clean complete."

clean-all: clean
	rm -rf $(LOG_DIR)/ $(RESULTS_DIR)/ $(WEIGHTS_DIR)/ data/processed/ data/tilm/
	@echo "Full clean complete (logs, results, weights removed)."
