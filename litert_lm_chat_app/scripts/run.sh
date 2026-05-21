#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp -n .env.example .env || true
echo "Edit .env and set LITERT_MODEL_PATH, then run: source .venv/bin/activate && python run.py"
