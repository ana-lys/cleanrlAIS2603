# takes in: previous_buffer_load_path, new_buffer_save_path, latest_replay_buffer_path, run_name
# training arguments too.
"""
Will then:
1. identify the list of training buffers from previous_buffer_load_path and latest_replay_buffer_path
2. Set up a dataset and dataloader that samples from the identified buffers
3. train a world model on the dataset and save it to new_buffer_save_path (include metadata about training buffers)
"""
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

from cleanrl.cleanrl_utils.port_gameboy_worlds import (
    WorldModel,
    get_pokeworlds_n_actions,
    PatchProjection,
    CNNEmbedder,
    PokemonReplayBuffer as ReplayBuffer,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    env_id: str = None
    """ id of the environment to get the action dimension and name tracking purposes """
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

    # Curiosity module specific arguments
    observation_embedder: str = "random_patch"
    """the type of observation embedder to use for the curiosity module."""
    embedder_load_path: str | None = None
    """path to load the observation embedder's weights from. Only applicable if the observation embedder supports loading."""
    buffer_load_path: str | None = None
    """ path to load the previous buffer metadata from (if None, only the latest will be used)"""
    latest_replay_buffer_folder: str | None = None
    """ folder to load the most replay buffer from """
    buffer_save_path: str | None = None
    """ path to save the current buffer metadata and trained world model """

    # World model training specific arguments
    num_epochs: int = 10
    """number of epochs to train the world model for"""
    batch_size: int = 64
    """ batch size to use for training the world model"""
    learning_rate: float = 1e-3
    """ learning rate to use for training the world model"""
    weight_decay: float = 1e-5
    """ weight decay to use for training the world model """
    early_stopping_patience: int = 3
    """ number of epochs with no improvement on the validation loss before stopping training early """
    force_overwrite: bool = False
    """ Skip training if model already exists if this is False """


class WorldModelDataset(Dataset):
    def __init__(self, args):
        self.buffer_paths = self.get_buffer_paths(args)
        self.buffer_lengths = self.create_dataset(args)
        self.cumulative_buffer_lengths = np.cumsum(self.buffer_lengths)
        self.total_length = sum(self.buffer_lengths)

    def __getitem__(self, idx):
        # figure out which buffer this index corresponds to
        buffer_idx = np.searchsorted(self.cumulative_buffer_lengths, idx, side="right")
        if buffer_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_buffer_lengths[buffer_idx - 1]
        buffer_path = self.buffer_paths[buffer_idx]
        X = torch.load(os.path.join(buffer_path, "X.pt"), mmap=True)
        y = torch.load(os.path.join(buffer_path, "y.pt"), mmap=True)
        return X[sample_idx].to("cuda"), y[sample_idx].to("cuda")

    def __len__(self):
        return self.total_length

    def get_buffer_paths(self, args):
        all_buffer_paths = []
        if args.latest_replay_buffer_folder:
            contents = os.listdir(args.latest_replay_buffer_folder)
            if (
                "observations.npy" in contents
            ):  # then its a run folder with a single replay buffer
                all_buffer_paths.append(args.latest_replay_buffer_folder)
            else:  # then its a folder containing multiple run folders, we want to use all of them
                for subfolder in contents:
                    subfolder_path = os.path.join(
                        args.latest_replay_buffer_folder, subfolder
                    )
                    if os.path.isdir(subfolder_path):
                        if "observations.npy" in os.listdir(subfolder_path):
                            all_buffer_paths.append(subfolder_path)
                        else:  # In case some run is ongoing. In general, we want to avoid this.
                            print(
                                f"Warning: {subfolder_path} does not contain a replay buffer, skipping..."
                            )
        if args.buffer_load_path:
            # previous_buffer_metadata is stored in buffer_load_path + "/buffer_metadata.txt" each line of this file is a path to a replay buffer that we will use for training the world model
            with open(
                os.path.join(args.buffer_load_path, "buffer_metadata.txt"), "r"
            ) as f:
                for line in f:
                    buffer_path = line.strip()
                    if buffer_path != "" and buffer_path not in all_buffer_paths:
                        all_buffer_paths.append(buffer_path)
        # validate that all buffer paths exist and contain the necessary files
        for buffer_path in all_buffer_paths:
            assert os.path.exists(
                buffer_path
            ), f"Buffer path {buffer_path} does not exist"
            assert os.path.exists(
                os.path.join(buffer_path, "observations.npy")
            ), f"Buffer path {buffer_path} does not contain observations.npy"
            assert os.path.exists(
                os.path.join(buffer_path, "actions.npy")
            ), f"Buffer path {buffer_path} does not contain actions.npy"
            assert os.path.exists(
                os.path.join(buffer_path, "last_step_indices.npy")
            ), f"Buffer path {buffer_path} does not contain last_step_indices.npy"
        return all_buffer_paths

    def get_embedder(self, args):
        if args.observation_embedder == "random_patch":
            observation_embedder = PatchProjection(normalized_observations=True).to(
                "cuda"
            )
        elif args.observation_embedder == "cnn":
            observation_embedder = CNNEmbedder(normalized_observations=True).to("cuda")
            # TODO: Figure out loading later.
        else:
            raise ValueError(
                f"Invalid observation embedder type: {args.observation_embedder}"
            )
        return observation_embedder

    def create_dataset(self, args):
        buffer_paths = self.buffer_paths
        embedder = self.get_embedder(args)
        action_dim = get_pokeworlds_n_actions(args.env_id)
        buffer_lengths = []
        for buffer_path in tqdm(buffer_paths, desc="Preprocessing buffers"):
            last_step_file = os.path.join(buffer_path, "last_step_indices.npy")
            last_steps = np.load(last_step_file, mmap_mode="r")
            observation_file = os.path.join(buffer_path, "observations.npy")
            actions_file = os.path.join(buffer_path, "actions.npy")
            observations = np.load(observation_file, mmap_mode="r")
            actions = np.load(actions_file, mmap_mode="r")
            X = []
            y = []
            for i in range(len(observations)):
                if i in last_steps:
                    continue  # skip the last step of each episode since we don't have a next observation for it
                obs = observations[i]
                action = actions[i].item()
                if i + 1 < len(observations):
                    next_obs = observations[i + 1][
                        0, -1
                    ]  # Get the most recent frame from the obs stack. Assumes a single env.
                else:
                    next_obs = observations[0][0, -1]
                obs_tensor = embedder.embed(obs).reshape(-1).to("cpu")
                action_tensor = torch.zeros(action_dim, dtype=torch.float16)
                action_tensor[action] = 1.0  # one-hot encode the action
                next_obs_tensor = embedder.embed(next_obs).to("cpu").reshape(-1)
                X_vec = torch.cat([obs_tensor, action_tensor])
                X.append(X_vec)
                y.append(next_obs_tensor)
            X = torch.stack(X).cpu()
            y = torch.stack(y).cpu()
            torch.save(X, buffer_path + "/X.pt")
            torch.save(y, buffer_path + "/y.pt")
            buffer_lengths.append(len(y))
            del X
            del y
        return buffer_lengths

    def close(self):
        for path in self.buffer_paths:
            os.remove(os.path.join(path, "X.pt"))
            os.remove(os.path.join(path, "y.pt"))

    def save_meta(self, save_path):
        with open(os.path.join(save_path, "buffer_metadata.txt"), "w") as f:
            for buffer_path in self.buffer_paths:
                f.write(buffer_path + "\n")
        print(
            f"Saved buffer metadata to {os.path.join(save_path, 'buffer_metadata.txt')}"
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.buffer_save_path is not None, "buffer_save_path must be specified"
    if os.path.exists(args.buffer_save_path + "/world_model.pt"):
        if not args.force_overwrite:
            print(
                f"World model already exists at {args.buffer_save_path}/world_model.pt, skipping training. Use --force_overwrite to overwrite."
            )
            exit(0)
        else:
            print(
                f"World model already exists at {args.buffer_save_path}/world_model.pt, but --force_overwrite is True, so overwriting..."
            )
    assert (
        args.buffer_load_path is not None
        or args.latest_replay_buffer_folder is not None
    ), "At least one of buffer_load_path or latest_replay_buffer_folder must be specified"
    if args.buffer_load_path is not None:
        assert os.path.exists(
            args.buffer_load_path
        ), f"buffer_load_path {args.buffer_load_path} does not exist"
    if args.latest_replay_buffer_folder is not None:
        assert os.path.exists(
            args.latest_replay_buffer_folder
        ), f"latest_replay_buffer_folder {args.latest_replay_buffer_folder} does not exist"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    assert torch.cuda.is_available(), "cuda flag is True but no cuda available"
    device = torch.device("cuda")
    dataset = WorldModelDataset(args)
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
    world_model = WorldModel(
        embedder=dataset.get_embedder(args),
        env_id=args.env_id,
    ).to(device)
    del (
        world_model.embedder
    )  # don't need this, have precomputed embeddings in the dataset
    optimizer = optim.Adam(
        world_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    run_name = f"{args.exp_name}_{args.buffer_save_path.replace('/', '--').replace('\\', '--')}_{int(time.time())}"
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
    for epoch in tqdm(range(args.num_epochs), desc="Training world model"):
        train_losses = []
        for X_batch, y_batch in tqdm(
            train_dataloader, desc="Epoch {epoch} - Training", leave=False
        ):
            optimizer.zero_grad()
            pred_next_obs = world_model(X_batch)
            loss = F.mse_loss(pred_next_obs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item() * len(X_batch))
        writer.add_scalar("train/loss_mean", np.mean(train_losses), epoch)
        writer.add_scalar("train/loss_std", np.std(train_losses), epoch)
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in tqdm(
                val_dataloader, desc="Epoch {epoch} - Validation", leave=False
            ):
                pred_next_obs = world_model(X_batch)
                loss = F.mse_loss(pred_next_obs, y_batch)
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
    # save the world model
    os.makedirs(args.buffer_save_path, exist_ok=True)
    torch.save(
        world_model.state_dict(), os.path.join(args.buffer_save_path, "world_model.pt")
    )
    dataset.save_meta(args.buffer_save_path)
    print(f"Saved world model and buffer metadata to {args.buffer_save_path}")
    dataset.close()
