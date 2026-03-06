#!/bin/bash
set -euo pipefail

CKPT_ROOT=${CKPT_ROOT:-"$SCRATCH/ckpts/mnist_causaldae"}
mkdir -p "$CKPT_ROOT"

RESUME_CKPT=${RESUME_CKPT:-"$CKPT_ROOT/latest.pt"}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}

# 建议用 srun 而不是 mpiexec（更贴合 Slurm 资源绑定）
CMD=(srun -n 1 python ./scripts/image_train.py
  --data_dir ./datasets/morphomnist
  --n_vars 2
  --in_channels 1
  --image_size 28
  --num_channels 128
  --num_res_blocks 3
  --causal_modeling True
  --learn_sigma False
  --class_cond True
  --rep_cond True
  --flow_based False
  --masking True
  --diffusion_steps 1000
  --noise_schedule linear
  --rescale_learned_sigmas False
  --rescale_timesteps False
  --lr 1e-5
  --batch_size 16
  --save_interval "$SAVE_INTERVAL"
  --use_checkpoint True
)

if [[ -f "$RESUME_CKPT" ]]; then
  CMD+=(--resume_checkpoint "$RESUME_CKPT")
  echo "[train] Resuming from $RESUME_CKPT"
else
  echo "[train] No checkpoint at $RESUME_CKPT, start fresh"
fi

echo "[train] CKPT_ROOT=$CKPT_ROOT"
echo "[train] save_interval=$SAVE_INTERVAL"
echo "[train] Running:"
printf " %q" "${CMD[@]}"; echo

"${CMD[@]}"