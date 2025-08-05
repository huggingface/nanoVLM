#!/bin/bash
#SBATCH --job-name=train_nanoVLM_torchrun
#SBATCH --output=logs/train_nanoVLM/%A.out
#SBATCH --error=logs/train_nanoVLM/%A.err
#SBATCH --time=47:59:00
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --partition=hopper-prod
#SBATCH --qos=high

module load cuda/12.9


export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_SOCKET_IFNAME=enp

export FI_EFA_ENABLE_SHM_TRANSFER=0
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# Change to project directory
cd /fsx/luis_wiedmann/nanoVLM
source .venv/bin/activate

# Activate virtual environment
export TOKENIZERS_PARALLELISM=false

# --- stage the venv on each node ------------------------------------------------
# Copy venv to scratch (once per node)
# srun --ntasks-per-node=1 bash -c '
#     echo "[$(hostname)] staging venv..."
#     rsync -a /fsx/luis_wiedmann/nanoVLM/nanoVLM_venv.tar.zst /scratch/
#     tar -xf /scratch/nanoVLM_venv.tar.zst -C /scratch/
#     rm /scratch/nanoVLM_venv.tar.zst
#     echo "venv staged"
# '


# Get the master node's address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# From https://i.hsfzxjy.site/2021-03-10-obtain-a-random-unused-tcp-port-with-bash/
function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
export MASTER_PORT=$(unused_port)

# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=120          # shorten watchdog; fails fast
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Run using torchrun on all allocated nodes
ulimit -n 99999
srun torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py