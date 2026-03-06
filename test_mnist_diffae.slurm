#!/bin/bash
#SBATCH --job-name=test_morphomnist_cf
#SBATCH --nodes=1
#SBATCH --partition=a10
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=00:30:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=test_morphomnist_cf.%j.out

module purge
module load external
module load anaconda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate improved-diffusion-clean

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH

cd /home/yu1331/CausalDiffAE

python scripts/image_diffae_test_morpho.py \
  --data_dir ./datasets/morphomnist \
  --model_path /scratch/gilbreth/yu1331/ckpts/mnist_causaldae/model021500.pt \
  --n_vars 2 \
  --in_channels 1 \
  --image_size 28 \
  --num_channels 128 \
  --num_res_blocks 3 \
  --causal_modeling True \
  --learn_sigma False \
  --class_cond True \
  --rep_cond True \
  --flow_based False \
  --masking True \
  --diffusion_steps 1000 \
  --noise_schedule linear \
  --rescale_learned_sigmas False \
  --rescale_timesteps False \
  --batch_size 16 \
  --out_dir /scratch/gilbreth/yu1331/results/morphomnist_cf \
  --intervene_idx 1
  --intervene_value 4.0