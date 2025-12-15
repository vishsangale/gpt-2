#!/bin/bash
set -e

# Common args
ARGS="--init_from=scratch --max_iters=100 --log_interval=10 --eval_interval=100 --always_save_checkpoint=True --batch_size=16"

echo "=== Running Vanilla Experiment ==="
venv/bin/python3 train.py $ARGS --out_dir=experiments/vanilla --use_rope=False --use_swiglu=False --use_rmsnorm=False

echo "=== Running RoPE Experiment ==="
venv/bin/python3 train.py $ARGS --out_dir=experiments/rope --use_rope=True --use_swiglu=False --use_rmsnorm=False

echo "=== Running SwiGLU Experiment ==="
venv/bin/python3 train.py $ARGS --out_dir=experiments/swiglu --use_rope=False --use_swiglu=True --use_rmsnorm=False

echo "=== Running RMSNorm Experiment ==="
venv/bin/python3 train.py $ARGS --out_dir=experiments/rmsnorm --use_rope=False --use_swiglu=False --use_rmsnorm=True

echo "=== All Experiments Completed ==="
