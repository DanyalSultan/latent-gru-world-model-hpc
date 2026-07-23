import os
import numpy as np
import torch
import torch.nn as nn
import wandb

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)

import yaml

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Configuration

with open("configs/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

wandb.init(
    project="latent-gru-world-model",
    name=f"latent-gru-train-hpc-a100-{os.environ.get('SLURM_JOB_ID', 'local')}",
    config=config
)

SEQ_LEN = 20
STATE_DIM = 77
ACTION_DIM = 7
INPUT_DIM = STATE_DIM + ACTION_DIM

LATENT_DIM = config["model"]["latent_dim"]
GRU_HIDDEN_DIM = config["model"]["gru_hidden_dim"]

BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["learning_rate"]

RECON_WEIGHT = config["loss"]["recon_weight"]
DYNAMICS_WEIGHT = config["loss"]["dynamics_weight"]

MODEL_PATH = "models/latent_gru_world_model.pt"

# Load dataset
X = np.load("dataset/X_latent_seq.npy").astype(np.float32)
Y = np.load("dataset/Y_latent_next_state.npy").astype(np.float32)

print("X:", X.shape)
print("Y:", Y.shape)

# Normalize state-action inputs and state targets
X_mean = X.mean(axis=(0, 1))
X_std = X.std(axis=(0, 1)) + 1e-8

Y_mean = Y.mean(axis=0)
Y_std = Y.std(axis=0) + 1e-8

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

os.makedirs("models", exist_ok=True)

np.save("models/latent_gru_X_mean.npy", X_mean)
np.save("models/latent_gru_X_std.npy", X_std)

np.save("models/latent_gru_Y_mean.npy", Y_mean)
np.save("models/latent_gru_Y_std.npy", Y_std)

# Dataset split
X_tensor = torch.tensor(X_norm)
Y_tensor = torch.tensor(Y_norm)

dataset = TensorDataset(X_tensor, Y_tensor)

N = len(dataset)

train_size = int(0.70 * N)
val_size = int(0.15 * N)
test_size = N - train_size - val_size

train_set, val_set, test_set = random_split(dataset,[train_size, val_size, test_size])

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Model
class LatentGRUWorldModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_DIM)
        )

        self.gru = nn.GRU(
            input_size=LATENT_DIM + ACTION_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=2,
            batch_first=True
        )

        self.latent_head = nn.Sequential(
            nn.Linear(GRU_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, LATENT_DIM)
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, STATE_DIM)
        )

    def forward(self, x):
        # x: [batch, sequence_length, 84]
        states = x[:, :, :STATE_DIM]
        actions = x[:, :, STATE_DIM:]

        # Encode every state in the history.
        z_history = self.encoder(states)

        # Feed latent-state/action pairs to GRU.
        gru_input = torch.cat([z_history, actions], dim=-1)
        gru_out, _ = self.gru(gru_input)

        # Final GRU output summarizes the 20-step history.
        final_hidden = gru_out[:, -1, :]

        # Predicted latent representation of next state.
        z_next_pred = self.latent_head(final_hidden)

        # Decode predicted next latent state.
        next_state_pred = self.decoder(z_next_pred)

        # Reconstruct the final observed state from its latent code.
        z_last_true = z_history[:, -1, :]
        last_state_recon = self.decoder(z_last_true)

        return (
            next_state_pred,
            z_next_pred,
            z_last_true,
            last_state_recon
        )
    

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = LatentGRUWorldModel().to(device)

wandb.config.update({
    "device": str(device),
    "parameters": sum(p.numel() for p in model.parameters())
})

mse = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = np.inf

# Training loop
for epoch in range(EPOCHS):
    model.train()

    train_total_loss = 0.0
    train_prediction_loss = 0.0
    train_latent_loss = 0.0
    train_reconstruction_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        (next_state_pred, z_next_pred, z_last_true, last_state_recon) = model(x_batch)

        # Main supervised next-state prediction loss.
        prediction_loss = mse(next_state_pred, y_batch)

        # Latent consistency: predicted next latent should match
        # the encoder's latent representation of the true next state.
        z_next_true = model.encoder(y_batch)

        latent_loss = mse(z_next_pred, z_next_true)

        # Autoencoder reconstruction of the final state in history.
        last_state_true = x_batch[:, -1, :STATE_DIM]

        reconstruction_loss = mse(last_state_recon, last_state_true)

        loss = (prediction_loss + DYNAMICS_WEIGHT * latent_loss + RECON_WEIGHT * reconstruction_loss)

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_total_loss += loss.item()
        train_prediction_loss += prediction_loss.item()
        train_latent_loss += latent_loss.item()
        train_reconstruction_loss += reconstruction_loss.item()

    train_total_loss /= len(train_loader)
    train_prediction_loss /= len(train_loader)
    train_latent_loss /= len(train_loader)
    train_reconstruction_loss /= len(train_loader)

    # Validation
    model.eval()

    val_total_loss = 0.0
    val_prediction_loss = 0.0
    val_latent_loss = 0.0
    val_reconstruction_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            (next_state_pred, z_next_pred, z_last_true, last_state_recon) = model(x_batch)

            prediction_loss = mse(next_state_pred, y_batch)

            z_next_true = model.encoder(y_batch)

            latent_loss = mse(z_next_pred, z_next_true)

            last_state_true = x_batch[:, -1, :STATE_DIM]

            reconstruction_loss = mse(last_state_recon, last_state_true)

            loss = (prediction_loss + DYNAMICS_WEIGHT * latent_loss + RECON_WEIGHT * reconstruction_loss)

            val_total_loss += loss.item()
            val_prediction_loss += prediction_loss.item()
            val_latent_loss += latent_loss.item()
            val_reconstruction_loss += reconstruction_loss.item()

    val_total_loss /= len(val_loader)
    val_prediction_loss /= len(val_loader)
    val_latent_loss /= len(val_loader)
    val_reconstruction_loss /= len(val_loader)

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Train {train_total_loss:.6f} | "
        f"Val {val_total_loss:.6f} | "
        f"Pred {val_prediction_loss:.6f} | "
        f"Latent {val_latent_loss:.6f} | "
        f"Recon {val_reconstruction_loss:.6f}"
    )

    wandb.log({
    "epoch": epoch + 1,
    "train_loss": train_total_loss,
    "val_loss": val_total_loss,
    "prediction_loss": val_prediction_loss,
    "latent_loss": val_latent_loss,
    "reconstruction_loss": val_reconstruction_loss
})

    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

wandb.save(MODEL_PATH)

print("\nBest validation loss:", best_val_loss)
print("Saved model:", MODEL_PATH)
wandb.finish()
