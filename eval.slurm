#!/bin/bash
#SBATCH --job-name=lmms_eval
#SBATCH --output=logs/lmms_eval/%j.out
#SBATCH --error=logs/lmms_eval/%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=11
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal

# Change to project directory
cd /fsx/luis_wiedmann/nanoVLM
source .venv/bin/activate

# Activate virtual environment
export TOKENIZERS_PARALLELISM=false

# Check if arguments are provided
if [ "$#" -ne 6 ]; then
    echo "Usage: sbatch eval.slurm <checkpoint_path> <global_step> <run_name> <limit> <tasks> <batch_size>"
    exit 1
fi

CHECKPOINT_PATH=$1
GLOBAL_STEP=$2
RUN_NAME=$3
LIMIT=$4
EVAL_TASKS=$5
EVAL_BATCH_SIZE=$6

echo "Starting evaluation for checkpoint: $CHECKPOINT_PATH at step $GLOBAL_STEP"

# Build base command
CMD="python run_evaluation.py \
    --checkpoint_path \"$CHECKPOINT_PATH\" \
    --global_step \"$GLOBAL_STEP\" \
    --run_name \"$RUN_NAME\" \
    --tasks \"$EVAL_TASKS\" \
    --batch_size \"$EVAL_BATCH_SIZE\""

# Only add limit if not None
if [ "$LIMIT" != "None" ]; then
    CMD="$CMD --limit \"$LIMIT\""
fi

eval $CMD

echo "Evaluation finished." 