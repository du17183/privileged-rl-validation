#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs"
if [[ -f "$ROOT/logs/launch_all.pid" ]]; then
  OLD_PID="$(cat "$ROOT/logs/launch_all.pid")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Experiment already running as PID $OLD_PID" >&2
    exit 1
  fi
fi
nohup bash "$ROOT/train/launch_all.sh" > "$ROOT/logs/launch_all.stdout.log" 2>&1 < /dev/null &
printf "%s\n" "$!" > "$ROOT/logs/launch_all.pid"
echo "Started 8-seed A/B/C experiment as PID $!"
