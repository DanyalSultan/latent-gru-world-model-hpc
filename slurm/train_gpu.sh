#!/bin/bash
#SBATCH --job-name=latentgru_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out

module load Anaconda3

source activate latentgru

cd /mnt/parscratch/users/aca22mds/projects/latent-gru-world-model/Latent-GRU-World-Model

echo "Running on:"
hostname

echo "CUDA check:"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

echo "Starting Latent GRU training..."

python scripts/train_latent_gru_world_model.py

if [ $? -eq 0 ]; then
    echo "Training finished successfully."
else
    echo "Training failed."
    exit 1
fi