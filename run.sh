#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Could not find virtual environment at $VENV_PATH" >&2
  echo "Create it first, then install requirements.txt." >&2
  exit 1
fi

cd "$SCRIPT_DIR"
source "$VENV_PATH/bin/activate"

exec "$VENV_PATH/bin/python" realsense_face_coordinate_udp.py --visualize "$@"
