#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_LINK_MODE=copy
UV="$(command -v uv)"
"$UV" venv --python "${PYTHON_BIN:-3.11}" --seed "$ROOT/.venv_pi05"
PY="$ROOT/.venv_pi05/bin/python"
"$UV" pip install --python "$PY" 'torch==2.14.0+cu130' 'torchvision==0.29.0+cu130' --index-url "${PI05_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
"$UV" pip install --python "$PY" -r "$ROOT/reproducibility/requirements-pi05-model.txt"
"$PY" - "$ROOT" <<'PY'
import pathlib, shutil, sys, transformers
root=pathlib.Path(sys.argv[1]); source=root/'reproducibility/transformers_overrides'
target=pathlib.Path(transformers.__file__).parent
assert transformers.__version__=='4.57.6'
for original in source.rglob('*.py'):
    destination=target/original.relative_to(source); destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(original,destination)
print('Copied exact recorded model sources into this new venv only.')
PY
"$PY" -m pip check
