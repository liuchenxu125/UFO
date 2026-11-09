#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"

# Persistent outputs and caches follow the project after its 2026-11-07 migration.
export UFO_CACHE_DIR="${UFO_CACHE_DIR:-$script_dir/cache/runtime_casbot02_20261106}"
# CLI options appended below override these defaults, including run name/size.
exec ./run_train.sh \
  --agent fb \
  --robot-config configs/robots/casbot02.yaml \
  --data-manifest configs/data/casbot02_lafan1_20261107_nopalms.yaml \
  --gpu-ids single \
  --num-envs 4096 \
  --num-env-steps 192000000 \
  --buffer-size 2560000 \
  --num-agent-updates 64 \
  --update-z-every-step 100 \
  --work-dir "$script_dir/runs/ufo_fb_casbot02_20261107_nopalms_4096" \
  "$@"
