#!/usr/bin/env bash
# Run from any directory. Override PYTHON_BIN to use a prepared environment.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
python_bin=${PYTHON_BIN:-$repo_dir/.venv/bin/python}
if [[ ! -x "$python_bin" ]]; then
  python_bin="$repo_dir/../../work/venv/bin/python"
fi
if [[ ! -x "$python_bin" ]]; then
  echo 'Prepare the environment with uv sync --frozen, or set PYTHON_BIN.' >&2
  exit 1
fi
export MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_dir/.cache/matplotlib}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$repo_dir/.cache/torchinductor}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Example: CUDA_VISIBLE_DEVICES=0 scripts/run_experiment.sh --run-name lafan_seed0
# All training options are passed directly to the typed training CLI.
if [[ ${NPROC_PER_NODE:-1} -gt 1 ]]; then
  exec "$python_bin" -m torch.distributed.run --standalone \
    --nproc_per_node "$NPROC_PER_NODE" -m supermimic.scripts.train "$@"
else
  exec "$python_bin" -m supermimic.scripts.train "$@"
fi
