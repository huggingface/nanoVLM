#!/bin/bash
#SBATCH --job-name=checkpoint_eval
#SBATCH --output=logs/checkpoint_eval/%A_%a.out
#SBATCH --error=logs/checkpoint_eval/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --array=0-14

# Change to project directory
cd /fsx/luis_wiedmann/nanoVLM
source .venv/bin/activate

tasks=(mmmu mmstar ocrbench textvqa chartqa ai2d mmbench_en mathvista hallusion_bench_image mmvet docvqa scienceqa mme infovqa seedbench)
runs=('checkpoints/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_13814samples_bs512_40000_lr5e-05-0.00512_0811-101603')
steps=(300)

#torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE run_checkpoint_evaluations.py --checkpoints_dir ${runs[@]} --eval_tasks ${tasks[@]} --steps ${steps[@]}
python run_checkpoint_evaluations.py --checkpoints_dir ${runs[@]} --eval_tasks ${tasks[$SLURM_ARRAY_TASK_ID]} --steps ${steps[@]}