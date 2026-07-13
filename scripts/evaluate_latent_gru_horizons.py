import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import wandb


# Configuration
import yaml

with open("configs/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

SEQ_LEN = config["model"]["seq_len"]
STATE_DIM = config["model"]["state_dim"]
ACTION_DIM = config["model"]["action_dim"]
LATENT_DIM = config["model"]["latent_dim"]
GRU_HIDDEN_DIM = config["model"]["gru_hidden_dim"]

EPISODE_DIR = "dataset/episode_001"

wandb.init(
    project="latent-gru-world-model",
    name="latent-gru-horizon-evaluation",
    config=config
)

# Model definition
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
        states = x[:, :, :STATE_DIM]
        actions = x[:, :, STATE_DIM:]

        z_history = self.encoder(states)

        gru_input = torch.cat([z_history, actions], dim=-1)
        gru_out, _ = self.gru(gru_input)

        final_hidden = gru_out[:, -1, :]

        z_next_pred = self.latent_head(final_hidden)
        next_state_pred = self.decoder(z_next_pred)
        return next_state_pred

# Load model and normalization statistics
model = LatentGRUWorldModel()

model.load_state_dict(
    torch.load(
        "models/latent_gru_world_model.pt",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

X_mean = np.load("models/latent_gru_X_mean.npy")
X_std = np.load("models/latent_gru_X_std.npy")

Y_mean = np.load("models/latent_gru_Y_mean.npy")
Y_std = np.load("models/latent_gru_Y_std.npy")

# Load episode
joint_pos = np.load(
    os.path.join(
        EPISODE_DIR,
        "joint_positions.npy"
    )
)

joint_vel = np.load(
    os.path.join(
        EPISODE_DIR,
        "joint_velocities.npy"
    )
)

eef_pos = np.load(
    os.path.join(
        EPISODE_DIR,
        "eef_positions.npy"
    )
)

eef_quat = np.load(
    os.path.join(
        EPISODE_DIR,
        "eef_quaternions.npy"
    )
)

object_state = np.load(
    os.path.join(
        EPISODE_DIR,
        "object_states.npy"
    )
)

actions = np.load(
    os.path.join(
        EPISODE_DIR,
        "actions.npy"
    )
)

states = np.concatenate(
    [
        joint_pos,
        joint_vel,
        eef_pos,
        eef_quat,
        object_state
    ],
    axis=1
).astype(np.float32)

print("State shape:", states.shape)

wandb.config.update({
    "episode": EPISODE_DIR,
    "timesteps": len(states)
})

# Predict one normalized next state from a sequence
def predict_next_state(sequence_raw):
    sequence_norm = (sequence_raw - X_mean) / X_std
    sequence_tensor = torch.tensor(sequence_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        next_state_norm = model(sequence_tensor)

    next_state_norm = (next_state_norm.cpu().numpy()[0])
    next_state = (next_state_norm * Y_std) + Y_mean
    return next_state

# Horizon evaluation

results = []

horizons = [1, 5, 10, 20, 50, 100]
for horizon in horizons:
    errors = []
    max_start = (len(states) - horizon - 1)

    for start in range(SEQ_LEN, max_start, 10):
        sequence = []
        
        for k in range(start - SEQ_LEN, start):
            state_action = np.concatenate([states[k], actions[k]])
            sequence.append(state_action)

        sequence = np.array(sequence, dtype=np.float32)
        current_state = states[start].copy()

        for h in range(horizon):
            next_state = predict_next_state(sequence)
            next_input = np.concatenate([next_state, actions[start + h]])
            sequence = np.vstack([sequence[1:], next_input])
            current_state = next_state

        true_state = states[start + horizon]
        rmse = np.sqrt(np.mean((current_state - true_state) ** 2))
        errors.append(rmse)

    mean_rmse = np.mean(errors)

    print(
        f"{horizon:3d}-step RMSE = "
        f"{mean_rmse:.6f}"
    )

    results.append(
        {
            "horizon": horizon,
            "rmse": mean_rmse
        }
    )

    wandb.log(
        {
            "horizon": horizon,
            "rmse": mean_rmse
        }
    )

os.makedirs("results", exist_ok=True)

df = pd.DataFrame(results)

wandb.log({
    "horizon_results": wandb.Table(
        dataframe=df
    )
})

df.to_csv(
    "results/horizon_rmse.csv",
    index=False
)

plt.figure(figsize=(8,5))

plt.plot(
    df["horizon"],
    df["rmse"],
    marker="o"
)

plt.xlabel("Prediction Horizon")
plt.ylabel("RMSE")
plt.title("Latent GRU World Model Long-Term Prediction Error")
plt.grid()

plt.savefig(
    "results/horizon_rmse.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

wandb.finish()
