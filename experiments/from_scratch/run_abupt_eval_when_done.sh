#!/usr/bin/env bash
# Wait for the AB-UPT from-scratch training process to finish, then evaluate the best
# (EMA) checkpoint through the UNCHANGED benchmark eval path (same dense eval config as the
# pretrained AB-UPT row) so the from-scratch test metrics are directly comparable. Then run
# the from-scratch-vs-pretrained aggregator.
#   bash experiments/from_scratch/run_abupt_eval_when_done.sh <train_pid> [gpu]
set -u
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/paint/bin/python
PID=${1:?usage: <train_pid> [gpu]}
GPU=${2:-0}
DS=${CARBENCH_SURFACE_DIR:-${CARBENCH_DATA_ROOT:-data/DrivAerNet}/PressureVTK/Dataverse}
SUB="${CARBENCH_SPLITS_DIR:-${CARBENCH_DATA_ROOT:-data/DrivAerNet}/train_val_test_splits}"
CKPT=results_from_scratch/abupt/abupt_from_scratch_best.th
LOG=results_from_scratch/abupt/eval.log

echo "[eval-watcher] waiting for AB-UPT train pid $PID ..."
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "[eval-watcher] training finished $(date)"
sleep 10
[ -f "$CKPT" ] || { echo "[eval-watcher] no checkpoint at $CKPT"; exit 1; }

echo "[eval-watcher] evaluating from-scratch AB-UPT (GPU $GPU) ..."
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_and_analyze.py abupt --evaluate-only \
  --dataset-path "$DS" --subset-dir "$SUB" \
  --config experiments/from_scratch/configs/abupt_eval.yaml \
  --checkpoint "$CKPT" --save-dir results_from_scratch/abupt \
  --enable-bootstrap > "$LOG" 2>&1
echo "[eval-watcher] eval done rc=$? $(date)"
$PY experiments/from_scratch/aggregate.py | tee results_from_scratch/abupt/comparison_snapshot.txt
echo "[eval-watcher] DONE"
