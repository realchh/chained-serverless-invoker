#!/bin/bash
set -euo pipefail

for tool in ruff mypy pytest; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing $tool; please install it in your environment."
    exit 1
  fi
done

PIP_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  PIP_FLAGS="--break-system-packages"
fi

# Install local packages (no network needed)
python -m pip install $PIP_FLAGS -e invoker -e middleware >/dev/null

echo "Running ruff..."
ruff --config invoker/pyproject.toml check invoker/chained_serverless_invoker
ruff --config middleware/pyproject.toml check middleware/serverless_tuner_middleware

echo "Running mypy..."
mypy --config-file invoker/pyproject.toml invoker/chained_serverless_invoker || python -m pip install $PIP_FLAGS types-requests
mypy --config-file invoker/pyproject.toml invoker/chained_serverless_invoker
mypy --config-file middleware/pyproject.toml middleware/serverless_tuner_middleware

echo "Running pytest..."
pytest
