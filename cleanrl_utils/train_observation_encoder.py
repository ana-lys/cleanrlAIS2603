import os
import random
import time
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro
from tqdm import tqdm

from cleanrl_utils.port_poke_worlds import (
    WorldModel,
    get_pokeworlds_n_actions,
    CNNEmbedder,
    PokemonReplayBuffer as ReplayBuffer,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Data arguments
    replay_buffer_folder: str | None = None
    """ folder that points to all the replay buffers for this game. Will train on every observation in the subdirectories of this folder """
    save_path: str | None = None
    """ path to save the trained observation encoder and metadata about the training buffers. """

    # Observation encoder training specific arguments
    num_epochs: int = 10
    """number of epochs to train the observation encoder for"""
    batch_size: int = 64
    """ batch size to use for training the observation encoder"""
    learning_rate: float = 1e-3
    """ learning rate to use for training the observation encoder"""
    weight_decay: float = 1e-5
    """ weight decay to use for training the observation encoder """
    early_stopping_patience: int = 3
    """ number of epochs with no improvement on the validation loss before stopping training early """
    force_overwrite: bool = True
    """ Skip training if model already exists if this is False """


class ObservationDataset(Dataset):
    def __init__(self, args):
        self.replay_buffer_folder = args.replay_buffer_folder
        self.observation_paths = self.get_all_observation_paths(
            self.replay_buffer_folder
        )
        buffer_lengths = self.create_dataset(args)
        self.cumulative_buffer_lengths = np.cumsum(buffer_lengths)
        self.total_length = self.cumulative_buffer_lengths[-1]

    def get_all_observation_paths(self, replay_buffer_folder):
        observation_paths = []
        for root, dirs, files in os.walk(replay_buffer_folder):
            for file in files:
                if file == "observations.npy":
                    observation_paths.append(os.path.join(root, file))
        return observation_paths

    def create_dataset(self, args):
        observation_files = self.observation_paths
        buffer_path = os.path.join(self.replay_buffer_folder, "processed_observations")
        os.makedirs(buffer_path, exist_ok=True)
        buffer_lengths = []
        n = 0
        for i, observation_file in tqdm(
            enumerate(observation_files), desc="Preprocessing buffers"
        ):
            observations = np.load(observation_file, mmap_mode="r")
            X = []
            for i in range(len(observations)):
                obs = observations[i][
                    -1
                ]  # Get the most recent frame from the obs stack. Assumes a single env.
                X_vec = torch.tensor(obs, dtype=torch.float16).reshape(-1).to("cpu")
                X.append(X_vec)
            X = torch.stack(X).cpu()
            torch.save(X, buffer_path + f"/X_{i}.pt")
            buffer_lengths.append(len(X))
            n += len(X)
            del X
        return buffer_lengths

    def __getitem__(self, idx):
        # figure out which buffer this index corresponds to
        buffer_idx = np.searchsorted(self.cumulative_buffer_lengths, idx, side="right")
        if buffer_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_buffer_lengths[buffer_idx - 1]
        X = torch.load(
            os.path.join(
                self.replay_buffer_folder,
                "processed_observations",
                f"X_{buffer_idx}.pt",
            ),
            mmap=True,
        )
        return X[sample_idx].to("cuda")

    def __len__(self):
        return self.total_length

    def close(self):
        os.remove(os.path.join(self.replay_buffer_folder, "processed_observations"))

    def save_meta(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "buffer_metadata.txt"), "w") as f:
            for buffer_path in self.observation_paths:
                f.write(buffer_path + "\n")
        print(
            f"Saved buffer metadata to {os.path.join(save_path, 'buffer_metadata.txt')}"
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert (
        args.replay_buffer_folder is not None
    ), "replay_buffer_folder must be specified"
    assert args.save_path is not None, "save_path must be specified"
    assert os.path.exists(
        args.replay_buffer_folder
    ), f"replay_buffer_folder {args.replay_buffer_folder} does not exist"
    if (
        os.path.exists(os.path.join(args.save_path, "observation_encoder.pt"))
        and not args.force_overwrite
    ):
        print(
            f"Model already exists at {os.path.join(args.save_path, 'observation_encoder.pt')}. Skipping training since force_overwrite is False."
        )
        exit(0)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    assert torch.cuda.is_available(), "cuda flag is True but no cuda available"
    device = torch.device("cuda")
    dataset = ObservationDataset(args)
    # split into train and val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    observation_embedder = CNNEmbedder().to(device).train()
    optimizer = optim.Adam(
        observation_embedder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_name = f"{args.exp_name}_{args.replay_buffer_folder.replace('/', '--').replace('\\', '--')}_{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    early_stopping_counter = 0
    best_val_loss = float("inf")
    for epoch in tqdm(range(args.num_epochs), desc="Training observation encoder"):
        train_losses = []
        for X_batch in tqdm(
            train_dataloader, desc="Epoch {epoch} - Training", leave=False
        ):
            optimizer.zero_grad()
            pred_next_obs = observation_embedder(X_batch)
            with torch.no_grad():
                y = observation_embedder.norm1(X_batch)
            loss = F.mse_loss(pred_next_obs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item() * len(X_batch))
        writer.add_scalar("train/loss_mean", np.mean(train_losses), epoch)
        writer.add_scalar("train/loss_std", np.std(train_losses), epoch)
        val_losses = []
        with torch.no_grad():
            for X_batch in tqdm(
                val_dataloader, desc="Epoch {epoch} - Validation", leave=False
            ):
                pred_next_obs = observation_embedder(X_batch)
                y = observation_embedder.norm1(X_batch)
                loss = F.mse_loss(pred_next_obs, y)
                val_losses.append(loss.item() * len(X_batch))
        avg_val_loss = np.mean(val_losses)
        writer.add_scalar("val/loss_mean", avg_val_loss, epoch)
        writer.add_scalar("val/loss_std", np.std(val_losses), epoch)
        # early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} with best validation loss {best_val_loss}"
                )
                break
    # save the observation embedder and metadata about the training buffers
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(
        observation_embedder.state_dict(),
        os.path.join(args.save_path, "observation_encoder.pt"),
    )
    dataset.save_meta(args.save_path)
    print(f"Saved observation encoder and buffer metadata to {args.save_path}")
    dataset.close()
