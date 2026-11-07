#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=""
export ART_CPU_HARNESS=1
export ART_TREE="$PWD"
export PYTHONPATH="$PWD/dev/memory_fixed_slope/cpu_imports:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --no-sync "$@"
