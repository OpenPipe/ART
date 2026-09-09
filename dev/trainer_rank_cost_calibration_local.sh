#!/usr/bin/env bash
# Cost-model calibration: single-GPU cells (TP1 x CP1) on a local H200.
#
# Usage: dev/trainer_rank_cost_calibration_local.sh <evidence.jsonl> [repeat]
# Requires the managed Megatron runtime venv (megatron_runtime/.venv) and, for
# the Ellavox guardrail groups, the git-ignored corpus in dev/.
set -euo pipefail
evidence=${1:?evidence jsonl path}
repeat=${2:-8}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE=1 ART_MEGATRON_CONTEXT_PARALLEL_SIZE=1
export ART_MEGATRON_DATA_PARALLEL_SIZE=1 ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE=1
logdir="$(dirname "${evidence}")"

cell() {  # cell model layers [group]
  timeout 7200s megatron_runtime/.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=1 \
    dev/trainer_rank_landing_acceptance.py --phase cost-calibrate \
    --cell "$1" --model "$2" --layers "$3" --group "${4:-0}" --repeat "${repeat}" \
    --evidence "${evidence}" 2>&1 | tee "${logdir}/tp1-cp1-$1-$3-g${4:-0}.log" \
    | { grep -E "cost-calibrate|LANDING|Traceback" || true; }
}
# pipefail carries torchrun's status through tee and the grep filter; a failed
# cell is recorded and the script exits nonzero at the end.
failures=0
run_cell() { cell "$@" || { failures=$((failures + 1)); echo "CELL FAILED: $*"; }; }
run_cell cal-grpo-g8-long Qwen/Qwen3.5-4B 2
run_cell cal-grpo-g8      Qwen/Qwen3.5-4B 0
run_cell cal-grpo-g16     Qwen/Qwen3.5-4B 0
run_cell cal-grpo-g4x4    Qwen/Qwen3.5-4B 0
run_cell cal-hetero       Qwen/Qwen3.5-4B 0
run_cell cal-hetero2      Qwen/Qwen3.5-4B 0
run_cell cal-hetero3      Qwen/Qwen3.5-4B 0
run_cell cal-grpo-g8-long Qwen/Qwen3-4B   2
run_cell cal-grpo-g8      Qwen/Qwen3-4B   0
run_cell cal-hetero2      Qwen/Qwen3-4B   0
if [ -f dev/_trainer_rank_ellavox_qwen35_4b_tokens.json ]; then
  for g in 0 1 2 3 4 5 6 7; do
    run_cell cal-ellavox Qwen/Qwen3.5-4B 0 "$g"
  done
fi
echo "cost-calibration local: done (${failures} failed cells)"
[ "${failures}" -eq 0 ]
