#!/bin/bash
#SBATCH --job-name=mnist_causaldae
#SBATCH --nodes=1
#SBATCH --partition=a10
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=job.%j.out
#SBATCH --mail-user=yu1331@purdue.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

set -eo pipefail

module load external
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate improved-diffusion-clean

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "[debug] SCRIPT_DIR=${SCRIPT_DIR:-<unset>}"

# 重新训练：换一个全新的 checkpoint 目录
export CKPT_ROOT="${SCRATCH:-$HOME}/ckpts/mnist_causaldae_newA"
mkdir -p "$CKPT_ROOT"

# 关键：不要 resume 旧 checkpoint
export RESUME_CKPT=""
export DIFFUSION_BLOB_LOGDIR="$CKPT_ROOT"

export SAVE_INTERVAL=2000

on_usr1() {
  echo "[`date`] Got SIGUSR1. Requeueing job $SLURM_JOB_ID ..."
  scontrol requeue "$SLURM_JOB_ID" || echo "WARN: requeue failed."
}
trap on_usr1 USR1

bash scripts/morhomnist/train_mnist_causaldae.sh
