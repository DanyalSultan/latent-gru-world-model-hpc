#!/bin/bash
#SBATCH --job-name=latentgru_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/pipeline_%j.out

module load Anaconda3
source activate latentgru

cd /mnt/parscratch/users/aca22mds/projects/latent-gru-world-model/Latent-GRU-World-Model

echo "Starting Latent GRU experiment"

echo "Training..."
python scripts/train_latent_gru_world_model.py

echo "Evaluation..."
python scripts/evaluate_latent_gru_horizons.py

echo "Rollout..."
python scripts/rollout_latent_gru_model.py

echo "Finished"
