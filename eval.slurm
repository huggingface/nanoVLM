#!/bin/bash
#SBATCH --job-name=lmms_eval
#SBATCH --output=logs/lmms_eval/%j.out
#SBATCH --error=logs/lmms_eval/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4                  # request as many GPUs as you want to use in parallel
#SBATCH --cpus-per-task=44            # total CPUs for the whole allocation
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --gres-flags=enforce-binding

set -euo pipefail

# Clean distributed defaults that can confuse eval scripts
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NCCL_SOCKET_IFNAME

cd /fsx/andi/nanoVLM
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

if [ "$#" -ne 6 ]; then
  echo "Usage: sbatch eval.slurm <checkpoint_path> <global_step> <run_name> <limit> <tasks> <batch_size>"
  exit 1
fi

CHECKPOINT_PATH=$1
GLOBAL_STEP=$2
RUN_NAME=$3
LIMIT=$4
EVAL_TASKS=$5          # comma-separated list, e.g. "mmstar,mmmu,ocrbench"
EVAL_BATCH_SIZE=$6

echo "Starting evaluation for checkpoint: $CHECKPOINT_PATH at step $GLOBAL_STEP"
echo "Tasks: $EVAL_TASKS"

# Discover available GPUs in this allocation
NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l | awk '{print $1}')}
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -lt 1 ]; then
  echo "No GPUs detected in allocation"
  exit 1
fi
echo "GPUs available: $NUM_GPUS"

# Compute CPU share per parallel worker
TOTAL_CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
CPUS_PER_WORKER=$(( TOTAL_CPUS / NUM_GPUS ))
if [ "$CPUS_PER_WORKER" -lt 1 ]; then CPUS_PER_WORKER=1; fi
echo "CPUs per worker: $CPUS_PER_WORKER (total: $TOTAL_CPUS)"

IFS=',' read -r -a TASK_ARR <<< "$EVAL_TASKS"

# Build the base common args once
BASE_ARGS=( run_evaluation.py
  --checkpoint_path "$CHECKPOINT_PATH"
  --global_step "$GLOBAL_STEP"
  --run_name "$RUN_NAME"
  --batch_size "$EVAL_BATCH_SIZE"
)
if [ "$LIMIT" != "None" ]; then
  BASE_ARGS+=( --limit "$LIMIT" )
fi

# Simple concurrency gate equal to number of GPUs
inflight=0
pids=()

for task in "${TASK_ARR[@]}"; do
  task_trimmed="$(echo "$task" | xargs)"
  if [ -z "$task_trimmed" ]; then
    continue
  fi

  echo "Launching task: $task_trimmed"

  # One srun per task, each grabs 1 GPU exclusively and CPUS_PER_WORKER CPUs
  srun --gres=gpu:1 --cpu-bind=cores --gpu-bind=closest -c "$CPUS_PER_WORKER" \
    python "${BASE_ARGS[@]}" --tasks "$task_trimmed" &
  pids+=( $! )
  inflight=$(( inflight + 1 ))

  # If we already launched as many tasks as GPUs, wait for one to finish
  if [ "$inflight" -ge "$NUM_GPUS" ]; then
    wait -n
    inflight=$(( inflight - 1 ))
  fi
done

# Wait for remaining tasks
if [ "${#pids[@]}" -gt 0 ]; then
  wait "${pids[@]}"
fi

# Merge per-task results into a single file for the step
echo "Merging results..."
python merge_eval_results.py --run_name "$RUN_NAME" --global_step "$GLOBAL_STEP"

echo "All evaluations finished and merged."
