#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec "$script_dir/.venv/bin/python" -m humanoidverse.sim2sim \
  --model-folder runs/ufo_fb_casbot02 \
  --robot-config configs/robots/casbot02.yaml \
  "$@"
