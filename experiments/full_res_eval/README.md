# Full-Resolution Evaluation (reviewer R1.1, round 2)

Three protocols per geometry on the official 1,154-car test split:
`sub10k` (10k prediction scored on those points), `interp` (upscaled to all
487,846 nodes; kNN-IDW at scale, RBF on a 61-car stratified subset to match the
manuscript's original protocol), `patch` (49 disjoint 10k passes, no interpolation).

    python run_full_res_eval.py --model M --gpu G [--interp knn|rbf] [--checkpoint CKPT]
    bash run_all.sh          # both sweeps over GPUs 1-3, resumable
    python measure_cost.py --gpu 1 --n-cars 20 --with-rbf
    python aggregate.py      # -> full_res_table.{csv,tex}

Validation gates (all passed): RBF reproduces the published single-car numbers
exactly (transolver 0.1499 sub / 0.2938 interp); patch matches single-pass on
shared points.

⚠ The app registry points `abupt`/`tripnet` at the FROM-SCRATCH checkpoints.
AB-UPT results here use `--checkpoint models/abupt/pressure_abupt_checkpoint.th`
(the author-released weights = AB-UPT* in the paper), tagged `_star`. No author
TripNet checkpoint exists on this machine; TripNet rows are the retrained model.
