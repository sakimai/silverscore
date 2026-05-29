#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOK_DIR="${ROOT_DIR}/notebooks"
NOTEBOOK_PATH="${NOTEBOOK_DIR}/paper_figures.ipynb"

echo "Running paper figures notebook..."
echo "Notebook: ${NOTEBOOK_PATH}"

python -m jupyter nbconvert \
  --to notebook \
  --execute "${NOTEBOOK_PATH}" \
  --inplace

echo "Done. Check outputs under ${NOTEBOOK_DIR}/figures"
