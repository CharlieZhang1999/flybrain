#!/usr/bin/env bash
# Reproducible setup: venv + deps + Shiu reference repo + FlyWire v783 data links.
# Run from the repo root: bash scripts/setup.sh
set -euo pipefail
unset PYTHONPATH  # a leaked PYTHONPATH makes pip skip packages and break the venv

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
if [ -f requirements.lock.txt ]; then
  .venv/bin/pip install -r requirements.lock.txt
else
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/pip check

mkdir -p third_party data/flywire_v783 data/malecns
if [ ! -d third_party/Drosophila_brain_model ]; then
  git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model third_party/Drosophila_brain_model
fi

# FlyWire v783 connectivity ships inside the Shiu repo (Zenodo 10676866 has the raw
# synapse tables, not these preprocessed files).
ln -sf ../../third_party/Drosophila_brain_model/Connectivity_783.parquet data/flywire_v783/Connectivity_783.parquet
ln -sf ../../third_party/Drosophila_brain_model/Completeness_783.csv data/flywire_v783/Completeness_783.csv

.venv/bin/python scripts/check_setup.py
