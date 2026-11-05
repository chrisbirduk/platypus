#!/usr/bin/env bash
# Container entry point: BUILD PRESET.env [SEED] [CONTROL.env ...]   (BUILD: platypus, the package)
set -euo pipefail
cd /work
nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "no GPU visible: run with --gpus all" >&2; exit 1; }
exec bash scripts/run_gpu.sh "$@"
