#!/bin/bash
set -e

# Common args
ARGS="--max_iters=1000 --log_interval=50 --eval_interval=200 --always_save_checkpoint=True --batch_size=16"

echo "=== Running All Features Enabled (Scratch, 1000 iters) ==="
# Note: Enabling all modernization flags
venv/bin/python3 train.py $ARGS --init_from=scratch --out_dir=experiments/all_features --use_rope=True --use_swiglu=True --use_rmsnorm=True

echo "=== Resuming Vanilla Baseline (Resume, up to 1000 iters) ==="
# Note: Resuming from experiments/vanilla (which is at 100 iters)
venv/bin/python3 train.py $ARGS --init_from=resume --out_dir=experiments/vanilla --use_rope=False --use_swiglu=False --use_rmsnorm=False

echo "=== Final Comparison Runs Completed ==="
