# Developer entry points. Windows users without GNU make can run the commands shown
# in each recipe directly, or use the container target (see README.md).

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
IMAGE ?= hypersat-processing-pipeline
TAG ?= dev

.DEFAULT_GOAL := help
.PHONY: help install install-gdal lint format type-check test test-integration test-external \
        check run-example docker-build docker-run clean

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in editable mode with development tooling
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install

install-gdal: ## Install the optional osgeo.gdal bindings (needs a matching system libgdal)
	$(PIP) install -e ".[dev,gdal]"

lint: ## Run Ruff lint and formatting checks
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format: ## Apply Ruff formatting and auto-fixable lint rules
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

type-check: ## Run mypy in strict mode
	$(PYTHON) -m mypy

test: ## Run unit tests (integration tests included, external products skipped)
	$(PYTHON) -m pytest

test-integration: ## Run only the integration tests
	$(PYTHON) -m pytest -m integration

test-external: ## Run tests that need real satellite products or DEMs (opt-in)
	$(PYTHON) -m pytest -m external

check: lint type-check test ## Run every quality gate, in the order CI uses

# Becomes functional in milestone 9, when `hypersat process` is implemented.
run-example: ## Run the example pipeline configuration
	$(PYTHON) -m hypersat.cli process --config configs/pipeline.example.yaml

docker-build: ## Build the container image
	docker build -t $(IMAGE):$(TAG) .

docker-run: ## Open a shell in the container with data/ and outputs/ mounted
	docker run --rm -it \
		-v "$(CURDIR)/data:/app/data" \
		-v "$(CURDIR)/outputs:/app/outputs" \
		-v "$(CURDIR)/configs:/app/configs:ro" \
		$(IMAGE):$(TAG) bash

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
