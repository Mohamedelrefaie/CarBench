#!/usr/bin/env bash
# Full paper-faithful TripNet from-scratch run (rebuttal point 2).
#   prep (parallel CPU) -> stage1 joint (1 GPU) -> stage2 on-the-fly (sharded GPUs) -> predictor
# Resumable: each stage skips work whose output already exists. Logs under $LOGDIR.
#
#   bash experiments/from_scratch/run_tripnet_full.sh
# Env overrides: STAGE1_GPU, STAGE2_GPUS, PRED_GPU, S1_EPOCHS, S2_EPOCHS, PRED_EPOCHS, PREP_SHARDS
set -u
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/paint/bin/python
TN=models/tripnet
L=experiments/from_scratch/tripnet_lists
DS="${CARBENCH_SURFACE_DIR:-${CARBENCH_DATA_ROOT:-data/DrivAerNet}/PressureVTK/Dataverse}"

CACHE=results_from_scratch/tripnet_cache
POOL=$CACHE/occ_pool                 # 500 stage1 pools (gitignored)
TRI=$CACHE/triplane_cache            # all 8121 triplanes + decoder (gitignored)
PRED_DIR=results_from_scratch/tripnet
LOGDIR=$PRED_DIR/logs
mkdir -p "$POOL" "$TRI" "$PRED_DIR" "$LOGDIR"

STAGE1_GPU=${STAGE1_GPU:-1}
STAGE2_GPUS=${STAGE2_GPUS:-"1 2 3"}
PRED_GPU=${PRED_GPU:-1}
S1_EPOCHS=${S1_EPOCHS:-500}
S2_EPOCHS=${S2_EPOCHS:-70}
PRED_EPOCHS=${PRED_EPOCHS:-200}
PREP_SHARDS=${PREP_SHARDS:-8}
NU=${NU:-1000000}; NS=${NS:-1000000}

echo "=== TripNet full run $(date) ==="
echo "stage1_gpu=$STAGE1_GPU stage2_gpus='$STAGE2_GPUS' pred_gpu=$PRED_GPU"
echo "epochs: s1=$S1_EPOCHS s2=$S2_EPOCHS pred=$PRED_EPOCHS | pool=${NU}+${NS}"

# ---------------- Phase 1: prep 500 stage1 pools (parallel CPU) ----------------
echo "=== [1/4] PREP stage1 pools ($(date)) ==="
pids=()
for i in $(seq 0 $((PREP_SHARDS-1))); do
  $PY $TN/occupancy_fit.py prep --cars-file $L/full_stage1.txt --out "$POOL" \
      --n-uniform "$NU" --n-surface "$NS" --shard "$i/$PREP_SHARDS" \
      > "$LOGDIR/prep_$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "prep done: $(ls "$POOL"/*.npz 2>/dev/null | wc -l) pools"

# ---------------- Phase 2: stage1 joint fit (1 GPU, resumable) ----------------
echo "=== [2/4] STAGE1 joint fit on GPU $STAGE1_GPU ($(date)) ==="
if [ ! -f "$TRI/decoder.pt" ]; then
  CUDA_VISIBLE_DEVICES=$STAGE1_GPU $PY $TN/occupancy_fit.py stage1 \
    --cars-file $L/full_stage1.txt --pooldir "$POOL" --out "$TRI" \
    --epochs "$S1_EPOCHS" --batch-pts 200000 --ckpt-every 25 --resume \
    > "$LOGDIR/stage1.log" 2>&1
else
  echo "decoder.pt exists -> skip stage1"
fi
[ -f "$TRI/decoder.pt" ] || { echo "STAGE1 FAILED (no decoder.pt)"; exit 1; }

# ---------------- Phase 3: stage2 on-the-fly (sharded GPUs, resumable) ----------------
echo "=== [3/4] STAGE2 on-the-fly, shards across GPUs '$STAGE2_GPUS' ($(date)) ==="
gpus=($STAGE2_GPUS); n=${#gpus[@]}; pids=()
for k in $(seq 0 $((n-1))); do
  g=${gpus[$k]}
  CUDA_VISIBLE_DEVICES=$g $PY $TN/occupancy_fit.py stage2 \
    --cars-file $L/full_stage2.txt --decoder "$TRI/decoder.pt" --out "$TRI" \
    --on-the-fly --epochs "$S2_EPOCHS" --batch-pts 200000 \
    --n-uniform "$NU" --n-surface "$NS" --shard "$k/$n" \
    > "$LOGDIR/stage2_shard${k}_gpu${g}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "stage2 done: $(ls "$TRI"/*.pt 2>/dev/null | grep -v decoder | wc -l) triplanes"

# ---------------- Phase 4: predictor train + test eval ----------------
echo "=== [4/4] PREDICTOR train+eval on GPU $PRED_GPU ($(date)) ==="
$PY experiments/from_scratch/train_tripnet.py \
  --tri-dir "$TRI" --train-cars $L/full_train.txt --test-cars $L/full_test.txt \
  --save-dir "$PRED_DIR" --gpu "$PRED_GPU" --epochs "$PRED_EPOCHS" \
  > "$LOGDIR/predictor.log" 2>&1
echo "=== TripNet full run COMPLETE $(date) ==="
cat "$PRED_DIR/tripnet_metrics.json" 2>/dev/null
