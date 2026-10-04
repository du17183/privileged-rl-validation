#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="$(command -v uv)"
"$UV" venv --python "${PYTHON_BIN:-3.11}" --seed "$ROOT/.venv"
"$UV" pip install --python "$ROOT/.venv/bin/python" \
  'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
"$UV" pip install --python "$ROOT/.venv/bin/python" \
  torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$ROOT/.venv/bin/python" \
  h5py tensorboard matplotlib

# Isaac Lab 2.3.0 still uses flatdict 4.0.1, whose build imports
# pkg_resources. Build it with the last setuptools series that provides it.
"$ROOT/.venv/bin/python" -m pip install setuptools==80.9.0
"$ROOT/.venv/bin/python" -m pip install --no-build-isolation flatdict==4.0.1

source "$ROOT/.venv/bin/activate"
cd "$ROOT/third_party/IsaacLab"
./isaaclab.sh --install none

# The upstream helper can continue after one extension's pip failure.
# Install the core explicitly and make dependency validation a hard gate.
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT/third_party/IsaacLab/source/isaaclab"
"$ROOT/.venv/bin/python" -m pip install \
  torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
"$ROOT/.venv/bin/python" -m pip install \
  wheel==0.44.0 packaging==23.0 click==8.1.7 psutil==5.9.8 \
  ipython==8.37.0 transformers==4.56.2 huggingface-hub==0.35.3
"$ROOT/.venv/bin/python" -m pip check

# CUDA 12.8 cannot JIT-compile the B300's SM 10.3 target. PyTorch's RPATH
# forces its own NVRTC, so replace only the NVRTC libraries inside this venv.
"$ROOT/.venv/bin/python" -m pip install --target "$ROOT/third_party/nvrtc129" \
  --no-deps nvidia-cuda-nvrtc-cu12==12.9.86
NVRTC_LIB="$ROOT/.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
mkdir -p "$ROOT/third_party/nvrtc128_backup"
cp -a -n "$NVRTC_LIB"/libnvrtc* "$ROOT/third_party/nvrtc128_backup/"
cp -a "$ROOT/third_party/nvrtc129/nvidia/cuda_nvrtc/lib"/libnvrtc* "$NVRTC_LIB/"

# The server lacks libGLU.so.1. Keep this Ubuntu runtime package project-local.
if [[ ! -e "$ROOT/third_party/system_libs/usr/lib/x86_64-linux-gnu/libGLU.so.1" ]]; then
  (cd "$ROOT/third_party" && apt download libglu1-mesa)
  dpkg-deb -x "$ROOT"/third_party/libglu1-mesa_*.deb "$ROOT/third_party/system_libs"
fi

cd "$ROOT"
"$ROOT/.venv/bin/python" envs/preflight.py
