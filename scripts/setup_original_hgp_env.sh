#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HGBO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${VENV_DIR:=$HGBO_ROOT/.venv113}"
: "${PYTHON_BOOTSTRAP:=$HGBO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BOOTSTRAP" ]]; then
  PYTHON_BOOTSTRAP=python3.9
fi

echo "Creating original HGBO-DSE HGP environment at $VENV_DIR"
"$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$HGBO_ROOT/requirements-original-hgp.txt"

"$VENV_DIR/bin/python" - <<'PY'
import torch
import torch_geometric
import torch_scatter
import torch_sparse

print("torch", torch.__version__)
print("torch_geometric", torch_geometric.__version__)
print("torch_scatter", torch_scatter.__version__)
print("torch_sparse", torch_sparse.__version__)
PY
