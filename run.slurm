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

# 固定 checkpoint 目录（放 scratch）
export CKPT_ROOT="${SCRATCH:-$HOME}/ckpts/mnist_causaldae"
mkdir -p "$CKPT_ROOT"
#export RESUME_CKPT="$CKPT_ROOT/latest.pt"
LATEST_CKPT=$(ls -1 "$CKPT_ROOT"/model*.pt 2>/dev/null | tail -n 1 || true)
export RESUME_CKPT="${LATEST_CKPT:-}"
export DIFFUSION_BLOB_LOGDIR="$CKPT_ROOT"

# 你可以按训练速度调小，让 kill 前一定能存到最新（比如 1000）
export SAVE_INTERVAL=2000

on_usr1() {
  echo "[`date`] Got SIGUSR1. Requeueing job $SLURM_JOB_ID ..."
  # 注意：这一步只是把 job 放回队列；当前这次运行仍会在 time limit 时结束
  scontrol requeue "$SLURM_JOB_ID" || echo "WARN: requeue failed."
}
trap on_usr1 USR1

bash scripts/morhomnist/train_mnist_causaldae.sh
