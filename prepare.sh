#!/bin/bash
#SBATCH --job-name=train_nanoVLM_torchrun
#SBATCH --output=logs/train_nanoVLM/%A_%a.out
#SBATCH --error=logs/train_nanoVLM/%A_%a.err
#SBATCH --time=47:59:00
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --array=4

echo "--- Starting parallel data copy on all nodes... ---"
# This srun command launches the copy script on all 4 nodes simultaneously.
# The shell will not proceed to the next line until ALL nodes have finished.
srun --ntasks-per-node=1 bash -c '
  mkdir -p /scratch/cache/the_cauldron && \
  cd /fsx/luis_wiedmann/.cache/the_cauldron && \
  find . -type f | parallel -j 16 rsync -R {} /scratch/cache/the_cauldron/
'
echo "--- All nodes have finished copying data. ---"

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
export NCCL_DEBUG=WARN

# Change to project directory
cd /fsx/luis_wiedmann/nanoVLM
source .venv/bin/activate

# Activate virtual environment
export TOKENIZERS_PARALLELISM=false

# -------------------------------------------------------------------------------

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

# Run using torchrun on all allocated nodes
ulimit -n 99999