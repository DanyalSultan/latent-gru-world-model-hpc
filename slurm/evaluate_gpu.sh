#!/bin/bash
#SBATCH --job-name=latentgru_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out

module load Anaconda3
module load CUDA/11.8.0

conda activate latentgru

cd /mnt/parscratch/users/aca22mds/projects/latent-gru-world-model/Latent-GRU-World-Model

python scripts/evaluate_latent_gru_horizons.py