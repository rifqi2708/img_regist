#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip

# In Codespaces, install directly into the current environment.
# If standard install is blocked, fallback to user site-packages.
python3 -m pip install -r requirements.txt || python3 -m pip install --user -r requirements.txt

echo "Setup complete."
