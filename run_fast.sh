#!/usr/bin/env bash
# Quick launcher that applies common performance envs and runs training with recommended flags.
# Usage: NGPU=4 ./run_fast.sh  # runs on 4 GPUs

set -euo pipefail

# recommended environment tweaks
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

# prefer to run with torchrun; default to 1 GPU if NGPU not set
NGPU=${NGPU:-1}

# set configs to use (change as needed)
CONFIGS="config/train_owt.py config/medium_model.py config/with_mhc_lite.py"

echo "Running with NGPU=${NGPU}, configs: ${CONFIGS}"

# pass compile flag via configurator override syntax
torchrun --standalone --nproc_per_node=${NGPU} train.py ${CONFIGS} --compile=True
