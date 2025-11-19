#!/bin/bash

set -e

# Run ruff
echo "Running ruff..."
poetry run ruff check --select I --fix .
poetry run ruff check --fix .
poetry run ruff format .

# Run mypy
echo "Running mypy..."
poetry run mypy --version
poetry run mypy .

# Run pytest
echo "Running pytest..."
poetry run pytest \
    --junitxml=pytest.xml \
    --cov-report=term-missing:skip-covered \
    --cov-fail-under=80 \
    --cov=chained_serverless_invoker \
    tests/