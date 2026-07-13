#!/bin/bash
#SBATCH --job-name=latentgru_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/train_%j.out

module load Anaconda3
module load CUDA/11.8.0

source activate latentgru

cd /mnt/parscratch/users/aca22mds/projects/latent-gru-world-model/Latent-GRU-World-Model

echo "Running on:"
hostname

echo "CUDA check:"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

python scripts/train_latent_gru_world_model.py
