#!/usr/bin/env bash
# Source this file before every Isaac Sim process on b300-2.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMNI_KIT_ACCEPT_EULA=yes
export ACCEPT_EULA=Y
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/third_party/system_libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
