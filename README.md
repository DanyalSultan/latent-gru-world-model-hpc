# Latent GRU World Model (HPC + W&B)

This repository contains a modified implementation of the **Latent GRU World Model** architecture, extended with a research-oriented training and evaluation pipeline.

The project explores latent-space recurrent world models for robot state prediction. The model learns a compact latent representation of robot dynamics and predicts future states from sequences of observations and actions.

This version introduces:

- Weights & Biases (W&B) experiment tracking
- GPU/HPC training support using SLURM
- Automated training and evaluation pipelines
- Config-based experiment management
- Multi-horizon rollout evaluation

---

# Modifications

Compared to the original Latent GRU implementation, this repository adds:

## Experiment Tracking

Integrated **Weights & Biases (W&B)** logging for:

- Training loss
- Validation loss
- Prediction loss
- Reconstruction loss
- Latent loss
- Rollout RMSE
- Long-horizon prediction metrics
- Model artifacts

---

## HPC Training Pipeline

Added SLURM-based GPU execution:

- Tested on University of Sheffield Stanage HPC
- NVIDIA A100 GPU support
- Automated experiment execution
- Separate training, rollout, and evaluation jobs

---

## Reproducibility Improvements

Added:

- YAML-based configuration
- Structured experiment scripts
- Consistent dataset preparation workflow
- Automated evaluation pipeline

---

# Repository Structure

```
Latent-GRU-World-Model/

│
├── configs/
│   └── train_config.yaml
│
├── scripts/
│   ├── build_latent_sequence_dataset.py
│   ├── train_latent_gru_world_model.py
│   ├── rollout_latent_gru_model.py
│   └── evaluate_latent_gru_horizons.py
│
├── slurm/
│   ├── train_gpu.sh
│   ├── rollout_gpu.sh
│   ├── evaluate_gpu.sh
│   └── run_experiment.sh
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Getting Started

There are two supported ways to run this project.

Choose the workflow that matches your environment.

---

# Option 1 — Local Execution

Use this if you are running on your own machine with a local GPU or CPU.

Recommended for:

- Development
- Debugging
- Small experiments
- Model testing

Follow:

[Local Setup](#local-setup)

---

# Option 2 — HPC Execution

Use this if you are running on a cluster environment with SLURM.

Recommended for:

- Large-scale experiments
- GPU training
- Long training runs
- Reproducible research experiments

Follow:

[HPC Setup](#hpc-setup-slurm)
---

# Local Setup

## 1. Create Environment

Create the conda environment:

```bash
conda create -n latentgru python=3.10
conda activate latentgru
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. W&B Setup

This project uses **Weights & Biases (W&B)** for experiment tracking.

Install:

```bash
pip install wandb
```

Login:

```bash
wandb login
```

After authentication, training and evaluation runs will automatically sync.

Tracked information:

- Training loss
- Validation loss
- Prediction loss
- Reconstruction loss
- Latent loss
- Rollout RMSE
- Long-horizon evaluation metrics
- Model checkpoints

---

## 3. Prepare Dataset

Before training, generate the sequence dataset:

```bash
python scripts/build_latent_sequence_dataset.py
```

This creates:

- Observation/action sequences
- Target future states
- Normalisation statistics

---

## 4. Train Model

Run:

```bash
python scripts/train_latent_gru_world_model.py
```

The best checkpoint is saved:

```
models/latent_gru_world_model.pt
```

---

## 5. Evaluate Model

Rollout evaluation:

```bash
python scripts/rollout_latent_gru_model.py
```

Long-horizon evaluation:

```bash
python scripts/evaluate_latent_gru_horizons.py
```

---

# HPC Setup (SLURM)

This workflow was tested on:

```
Platform:
University of Sheffield Stanage HPC

GPU:
NVIDIA A100-SXM4-80GB

CUDA:
12.4

PyTorch:
2.6.0+cu124

Python:
3.10
```

---

## 1. Load HPC Environment

Example:

```bash
module load Anaconda3
conda activate latentgru
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Configure W&B

W&B works the same way on HPC.

Login:

```bash
wandb login
```

Training jobs will automatically upload:

- Loss curves
- Metrics
- Configurations
- Checkpoints
- Evaluation results

---

## 3. Submit Full Pipeline

The recommended HPC workflow is:

```bash
sbatch slurm/run_experiment.sh
```

This executes:

```
Dataset Preparation
        ↓
Model Training
        ↓
Rollout Evaluation
        ↓
Long-Horizon Evaluation
```

---

## 4. Individual HPC Jobs

Individual stages can also be submitted separately.

### Dataset / Training

```bash
sbatch slurm/train_gpu.sh
```

### Rollout

```bash
sbatch slurm/rollout_gpu.sh
```

### Horizon Evaluation

```bash
sbatch slurm/evaluate_gpu.sh
```

---

# Experiment Workflow

Regardless of execution method, the model pipeline is:

```
Raw Dataset
      ↓
build_latent_sequence_dataset.py
      ↓
Processed Sequence Dataset
      ↓
train_latent_gru_world_model.py
      ↓
latent_gru_world_model.pt
      ↓
rollout_latent_gru_model.py
      ↓
evaluate_latent_gru_horizons.py
```

---

# Data and Generated Files

Large generated files are intentionally excluded from Git:

```
dataset/
models/
logs/
wandb/
```

These are recreated during execution.

The repository contains only:

- Source code
- Configuration files
- SLURM scripts
- Documentation

---

# Acknowledgements

Original Latent GRU World Model architecture developed by:

**Zaryan Ali Ansari**

This repository contains additional modifications for:

- HPC execution
- W&B experiment tracking
- Automated evaluation
- Research reproducibility
