#!/bin/bash
#SBATCH --job-name=checkpoint_eval
#SBATCH --output=logs/checkpoint_eval/%A_%a.out
#SBATCH --error=logs/checkpoint_eval/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --array=0

# Change to project directory
cd /fsx/luis_wiedmann/nanoVLM
source .venv/bin/activate

tasks='mmstar,textvqa,ocrbench,mmmu'
runs=('checkpoints/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_1988samples_bs512_40000_lr5e-05-0.00512_0808-150443')

python run_checkpoint_evaluations.py --checkpoints_dir ${runs[$SLURM_ARRAY_TASK_ID]} --eval_tasks $tasks
