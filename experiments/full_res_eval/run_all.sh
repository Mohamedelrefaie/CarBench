#!/usr/bin/env bash
# Full-resolution evaluation for reviewer R1.1, two sweeps over GPUs 1-3 (GPU 0 reserved):
#   Sweep 1 (all 1,154 test cars): sub10k + interp(kNN-IDW) + patch     — the headline numbers
#   Sweep 2 (61 stratified cars) : interp(RBF) = the manuscript's original upscaling
# Resumable: a model is skipped if its result JSON already exists.
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate
E=experiments/full_res_eval
LOG=$E/results/logs
mkdir -p "$LOG" "$E/results"

sweep1() {  # $1 model  $2 gpu
  [ -f "$E/results/$1.json" ] && { echo "skip $1 (sweep1 done)"; return 0; }
  python $E/run_full_res_eval.py --model "$1" --gpu "$2" --interp knn \
      > "$LOG/$1_sweep1.log" 2>&1 || echo "FAILED sweep1 $1"
}
sweep2() {  # $1 model  $2 gpu
  [ -f "$E/results/$1_rbf.json" ] && { echo "skip $1 (sweep2 done)"; return 0; }
  mapfile -t CARS < $E/rbf_subset.txt
  python $E/run_full_res_eval.py --model "$1" --gpu "$2" --interp rbf --skip-patch \
      --tag rbf --cars "${CARS[@]}" \
      > "$LOG/$1_sweep2.log" 2>&1 || echo "FAILED sweep2 $1"
}

lane() {  # $1 gpu, rest = models
  local gpu=$1; shift
  for m in "$@"; do sweep1 "$m" "$gpu"; done
  for m in "$@"; do sweep2 "$m" "$gpu"; done
}

lane 1 transolver transolver_large transolver_plus fno &
L1=$!
lane 2 abupt tripnet pointmae pointnet &
L2=$!
lane 3 regdgcnn point_transformer pointnet_large &
L3=$!
echo "lanes: $L1 $L2 $L3 — logs in $LOG/"
wait $L1 $L2 $L3
echo "ALL FULL-RES EVAL DONE"
