#!/usr/bin/env bash
# Create a virtualenv on first run, install deps, launch the app at http://localhost:8501
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
streamlit run app.py --server.address localhost --server.port 8501
