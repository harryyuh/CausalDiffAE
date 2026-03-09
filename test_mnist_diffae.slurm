#!/bin/bash
#SBATCH --job-name=test_morphomnist_cf
#SBATCH --nodes=1
#SBATCH --partition=a10
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00

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

# 新 checkpoint
MODEL=/scratch/gilbreth/yu1331/ckpts/mnist_causaldae_newA/model026000.pt

# 新 output 文件夹
OUT=/scratch/gilbreth/yu1331/results/morphomnist_cf_nonormal
mkdir -p $OUT

# Based on dataset summary statistics
# idx 0 = thickness
# idx 1 = slant
thickness_values=(1.1 2.0 2.6 4.0 7.0)
slant_values=(-0.32 -0.10 0.10 0.30 0.51)

echo "Testing thickness intervention (idx=0)"
for val in "${thickness_values[@]}"
do
  python scripts/image_diffae_test_morpho.py \
    --data_dir ./datasets/morphomnist \
    --model_path $MODEL \
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
    --out_dir ${OUT}/thickness_${val} \
    --intervene_idx 0 \
    --intervene_value $val
done

echo "Testing slant intervention (idx=1)"
for val in "${slant_values[@]}"
do
  python scripts/image_diffae_test_morpho.py \
    --data_dir ./datasets/morphomnist \
    --model_path $MODEL \
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
    --out_dir ${OUT}/slant_${val} \
    --intervene_idx 1 \
    --intervene_value $val
done