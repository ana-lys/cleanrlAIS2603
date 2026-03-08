from poke_worlds import get_environment
from poke_worlds.emulation import StateParser
import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
import torch
import torch.nn as nn
from typing import List
from sklearn.cluster import MiniBatchKMeans, KMeans
import os
import pickle
from cleanrl_utils.buffers import ReplayBuffer
from matplotlib import pyplot as plt


class MaxLengthList:
    def __init__(self, max_length):
        self.max_length = max_length
        self.data = []

    def insert(self, item, index):
        if index >= self.max_length:
            raise IndexError(
                f"Index {index} out of bounds for MaxLengthList with max_length {self.max_length}"
            )
        self.data.insert(index, item)
        if len(self.data) > self.max_length:
            self.data.pop(0)

    def get_insert_index(self, item):
        """
        Get the index where the item should be inserted to preserve a descending sorted order
        """
        for i, existing_item in enumerate(self.data):
            if item > existing_item:
                return i
        if len(self.data) < self.max_length:
            return len(self.data)
        return (
            None  # item is not greater than any existing item and list is at max length
        )

    def do_item_insert(self, item):
        index = self.get_insert_index(item)
        if index is not None:
            self.insert(item, index)
            return index
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)


def save_model(model_data, model_save_folder):
    os.makedirs(model_save_folder, exist_ok=True)
    model_save_path = os.path.join(model_save_folder, "model.pt")
    torch.save(model_data, model_save_path)
    print(f"model saved to {model_save_path}")


def save_ranked_models(model_data_list, model_save_folder):
    for i, model_data in enumerate(model_data_list):
        save_model(model_data, os.path.join(model_save_folder, f"rank_{i+1}"))


def save_all_models(final_model_data, model_data_list, model_save_folder):
    if model_save_folder is None:
        print(f"Warning: model_save_folder is None. Models will not be saved.")
        return
    save_model(final_model_data, os.path.join(model_save_folder, f"final"))
    save_ranked_models(model_data_list, model_save_folder)


def depathify(string):
    return (
        string.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


FRAME_STACK = 2


class OneOfToDiscreteWrapper(gym.ActionWrapper):
    STATIC_MAP = {}
    """ Set on init to allow static access of a dict mapping actions to HighLevelActions """

    def __init__(self, env):
        super().__init__(env)
        # Calculate total actions across all sub-spaces
        # Example: OneOf([Discrete(2), Discrete(3)]) -> total 5
        self.internal_env = env
        self.sub_spaces = env.action_space.spaces
        self.total_actions = sum(s.n for s in self.sub_spaces)
        self.action_space = Discrete(self.total_actions)
        for action in range(self.action_space.n):
            high_level_action, kwargs = self.get_high_level_action(action)
            OneOfToDiscreteWrapper.STATIC_MAP[action] = (high_level_action, kwargs)

    def action(self, action):
        # Map the single integer back to (choice, sub_action)
        offset = 0
        for i, space in enumerate(self.sub_spaces):
            if action < offset + space.n:
                return (i, action - offset)
            offset += space.n
        print("Action mapping error!")
        return (0, 0)  # Fallback

    def get_high_level_action(self, action):
        # Map the single integer back to choice only
        action = self.action(action)
        high_level_action, kwargs = (
            self.internal_env._controller._space_action_to_high_level_action(action)
        )
        return high_level_action, kwargs

    def set_render_mode(self, mode):
        self.internal_env.render_mode = mode

    @staticmethod
    def get_high_level_action_static(action):
        if len(OneOfToDiscreteWrapper.STATIC_MAP) == 0:
            raise ValueError("STATIC_MAP not initialized yet!")
        return OneOfToDiscreteWrapper.STATIC_MAP[action]


def parse_pokeworlds_id_string(id_string):
    """

    :param id_string: should be in format "poke_worlds-game-environment_variant-init_state-controller_variant-max_steps-save_video"
    Example: poke_worlds-pokemon_red-starter_explore-none-low_level-20-true
    :return: tuple (game, environment_variant, init_state, controller_variant, max_steps, save_video)
    """
    #
    parts = id_string.split("-")
    if len(parts) != 7 or parts[0] != "poke_worlds":
        raise ValueError(
            f"Invalid ID string format. Expected 'poke_worlds-game-environment_variant-init_state-controller_variant-max_steps-save_video'. Got {id_string}"
        )
    (
        _,
        game,
        environment_variant,
        init_state,
        controller_variant,
        max_steps_str,
        save_video_str,
    ) = parts
    if not max_steps_str.isdigit():
        raise ValueError(
            f"Invalid max_steps value. Expected an integer. Got {max_steps_str}"
        )
    max_steps = int(max_steps_str)
    save_video = save_video_str.lower() == "true"
    if init_state.lower() == "none":
        init_state = None
    return (
        game,
        environment_variant,
        init_state,
        controller_variant,
        max_steps,
        save_video,
    )


def get_poke_worlds_environment(id_string, render_mode=None):
    game, environment_variant, init_state, controller_variant, max_steps, save_video = (
        parse_pokeworlds_id_string(id_string)
    )
    env = get_environment(
        game=game,
        controller_variant=controller_variant,
        init_state=init_state,
        environment_variant=environment_variant,
        max_steps=max_steps,
        headless=True,
        save_video=save_video,
    )
    env = OneOfToDiscreteWrapper(env)
    if render_mode is not None:
        env.set_render_mode(render_mode)
    return env


def get_pokeworlds_n_actions(id_string=None):
    if len(OneOfToDiscreteWrapper.STATIC_MAP) == 0:
        if id_string is not None:
            _ = get_poke_worlds_environment(id_string)
        else:
            raise ValueError(
                f"STATIC_MAP not initialized yet! Please provide an id_string to initialize the environment and action mapping."
            )
    return len(OneOfToDiscreteWrapper.STATIC_MAP)


def poke_worlds_make_env(env_id, seed, idx, capture_video, run_name, gamma=0.99):
    def thunk():
        if capture_video and idx == 0:
            env = get_poke_worlds_environment(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = get_poke_worlds_environment(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(
            env, (144, 160)
        )  # Don't ask me why, but this is needed.
        env = gym.wrappers.FrameStackObservation(env, FRAME_STACK)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)

        if seed is not None:
            env.action_space.seed(seed)
        return env

    return thunk


class PatchProjection(nn.Module):
    """
    Works with the 144 x 160 pixel observations from poke_worlds.
    Divides the image into 16x16 patches, applies a random linear projection to each patch, and concatenates the results.
    """

    def __init__(self, normalized_observations=True):
        super().__init__()
        self.normalized_observations = normalized_observations
        self.project = nn.Sequential(
            nn.Conv2d(
                1,
                1,
                kernel_size=8,
                stride=8,  # 8x8 patches with no overlap to get 4 snapshots of each of the gameboys 16x16 cells.
            ),
            nn.Flatten(),
        )
        self.output_dim = 90 * 4
        self.dtype = self.project[0].weight.dtype

    def forward(self, x):
        vector = self.project(x)
        if self.normalized_observations:
            normalized = nn.functional.normalize(vector, dim=-1)
            return normalized
        return vector

    def embed(self, items: List[np.ndarray]) -> torch.Tensor:
        with torch.no_grad():
            if not isinstance(items, torch.Tensor):
                batch_tensor = torch.tensor(
                    items.reshape(-1, 1, 144, 160),
                )
            batch_tensor = batch_tensor.to(self.dtype).to(
                next(self.parameters()).device
            )
            embeddings = self(batch_tensor)
            return embeddings


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_gameboy_cnn_chain(stacked=True):
    use_stack = FRAME_STACK if stacked else 1
    return nn.Sequential(
        layer_init(
            nn.Conv2d(use_stack, 32, kernel_size=16, stride=16)
        ),  # (batch_size, 32, 9, 10)
        nn.ReLU(),
        layer_init(
            nn.Conv2d(32, 64, kernel_size=4, stride=2)
        ),  # (batch_size, 64, 3, 4)
        nn.ReLU(),
        layer_init(
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        ),  # (batch_size, 64, 1, 2)
        nn.ReLU(),
        nn.Flatten(),  # (batch_size, 128)
    )


def invert_gameboy_cnn_chain(stacked=True):
    use_stack = FRAME_STACK if stacked else 1
    return nn.Sequential(
        nn.Unflatten(1, (64, 1, 2)),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=(1, 0)),
        nn.ReLU(),
        nn.ConvTranspose2d(32, use_stack, kernel_size=16, stride=16),
    )


class CNNEmbedder(nn.Module):
    def __init__(self, hidden_dim=128, normalized_observations=True):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(1, affine=False)
        self.internal_norm = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm2d(1, affine=False)
        encoder_cnn_chain = get_gameboy_cnn_chain(stacked=False)
        dummy_input = torch.zeros(1, 1, 144, 160)
        with torch.no_grad():
            dummy_output = encoder_cnn_chain(dummy_input)
        chain_dim = dummy_output.shape[1]
        self.encoder = nn.Sequential(
            *get_gameboy_cnn_chain(stacked=False),
            layer_init(nn.Linear(chain_dim, hidden_dim)),
            nn.Sigmoid(),
            self.internal_norm,
        )
        self.decoder = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, chain_dim)),
            *invert_gameboy_cnn_chain(stacked=False),
        )
        self.output_dim = hidden_dim
        self.normalized_observations = normalized_observations

    def do_embed(self, x):
        normed = self.norm1(x)
        raw = self.encoder(normed)
        if self.normalized_observations:
            normalized = nn.functional.normalize(
                raw, dim=-1
            )  # Normalize the output embeddings
            return normalized
        return raw

    def forward(self, x):
        embedding = self.do_embed(x)
        unembed = self.decoder(embedding)
        normed = self.norm2(unembed)
        return normed

    def embed(self, items: List[np.ndarray]) -> torch.Tensor:
        with torch.no_grad():
            batch_tensor = torch.tensor(
                items.reshape(-1, 1, 144, 160),
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )
            embeddings = self.do_embed(batch_tensor)
            return embeddings

    def load(self, path):
        loaded_state = torch.load(path)
        self.load_state_dict(loaded_state)
        print(f"Loaded CNN embedder from {path}")


class WorldModel(nn.Module):
    def __init__(
        self,
        embedder,
        hidden_dim=512,
        normalized_observations=True,
        load_path=None,
        save_path=None,
        env_id=None,
    ):
        super().__init__()
        self.embedder = embedder
        self.model = None
        self.hidden_dim = hidden_dim
        self.normalized_observations = normalized_observations
        self.save_path = save_path
        self.load_path = load_path
        if env_id is not None:
            action_dim = get_pokeworlds_n_actions(env_id)
            self.create_model(action_dim)

    def create_model(self, action_dim=None):
        observation_dim = self.embedder.output_dim * FRAME_STACK
        if action_dim is None:
            action_dim = get_pokeworlds_n_actions()  # attempt to get from static map
        self.action_dim = action_dim
        hidden_dim = self.hidden_dim
        self.model = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedder.output_dim),
        )

    def forward(self, x):
        """x should be concatenated embedding and action tensor of shape (batch_size, observation_dim + 1)"""
        next_obs_pred = self.model(x)
        if self.normalized_observations:
            next_obs_pred = nn.functional.normalize(next_obs_pred, dim=-1)
        return next_obs_pred

    def predict(self, raw_obs, action):
        if self.model is None:
            self.create_model()
        with torch.no_grad():
            obs = self.embedder.embed(raw_obs).reshape(-1)  # flatten the frame stack
            action_vector = torch.zeros(
                self.action_dim, dtype=obs.dtype, device=obs.device
            )
            action_vector[action] = 1.0  # one-hot encode the action
            x = torch.cat([obs, action_vector], dim=-1)
            output = self.forward(x)
        return output

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        with torch.no_grad():
            next_obs = next_obs[
                0, -1
            ]  # get the last frame of the frame stack. THIS COMMITS TO ONLY ONE ENV
            next_obs_embed = self.embedder.embed(next_obs)
            predicted_next_obs_embed = self.predict(raw_obs=obs, action=actions)
            # reward is the error in the embedding space
            if self.normalized_observations:
                # reward is 1 - cosine_similarity. Since the embeddings are normalized, cosine similarity is just the dot product.
                reward = 1 - torch.dot(predicted_next_obs_embed, next_obs_embed).item()
            else:
                # MSE between the vectors
                reward = torch.mean(
                    (predicted_next_obs_embed - next_obs_embed) ** 2
                ).item()
        return reward

    def reset(self):
        if self.load_path is not None and self.model is None:
            self.load()

    def iterative_save(self):
        pass

    def load(self):
        self.create_model(
            action_dim=get_pokeworlds_n_actions()
        )  # this is safe because it is only called after the STATIC_MAP is initialized by creating an environment, which happens in the training loop before the world model is used.
        loaded_state = torch.load(self.load_path)
        self.model.load_state_dict(loaded_state)
        print(f"Loaded world model from {self.load_path}")


def get_passed_frames(infos) -> np.ndarray:
    # infos['core']['passed_frames'].shape == (1, n_frames, 144, 160, 1)
    frames = infos["core"]["passed_frames"]
    if len(frames.shape) == 1:  # then a reset has happened. must use current frame
        frames = infos["core"]["current_frame"]
    return frames.squeeze(0).reshape(-1, 144, 160)


class EmbedBuffer:
    def __init__(
        self,
        embedder,
        similarity_metric="cosine",
        load_path=None,
        save_path=None,
        max_size=10_000,
    ):
        self.max_size = max_size
        self.embedder = embedder
        self.save_path = save_path
        self.load_path = load_path
        if self.save_path is not None and self.save_path == self.load_path:
            print(
                f"Warning: save_path and load_path are the same. This means the buffer will be overwritten on reset and grow over time. This should only be used with a random agent to accumilate base observation data."
            )
        similarity_options = ["cosine", "distance", "hinge"]
        if similarity_metric not in similarity_options:
            raise ValueError(
                f"Invalid similarity metric {similarity_metric}. Must be one of {similarity_options}"
            )
        self.similarity_metric = similarity_metric
        self.buffer = None
        self.reset()

    def get_unseen_elements(self, embeddings, buffer=None):
        if buffer is None:
            buffer = self.buffer
        if buffer is None:
            return embeddings
        # embedding shape: (n_frames, embedding_dim)
        # buffer shape: (buffer_size, embedding_dim)
        diffs = embeddings.unsqueeze(1) - buffer.unsqueeze(0)
        new_embeddings = []
        for i in range(embeddings.shape[0]):
            max_dimension_diff = (
                diffs[i].abs().max(-1).values
            )  # max absolute difference across dimensions for each buffer element
            has_element_too_close = (
                max_dimension_diff.min().item() < 0.001
            )  # if any buffer element is too close in any dimension, we consider it already in the buffer
            if not has_element_too_close:
                new_embeddings.append(embeddings[i])
        if len(new_embeddings) == 0:
            return None
        return torch.stack(new_embeddings)

    def iterative_save(self):
        if self.save_path is not None and self.buffer is not None:
            os.makedirs(self.save_path, exist_ok=True)
            save_size = self.buffer.shape[0]
            if os.path.exists(self.save_path + "/embed_buffer.pt"):
                existing_buffer = torch.load(self.save_path + "/embed_buffer.pt").to(
                    next(self.embedder.parameters()).device
                )
                existing_buffer = self.get_unseen_elements(existing_buffer)
                if existing_buffer is not None:
                    merged_buffer = torch.cat([existing_buffer, self.buffer], dim=0)
                else:
                    print(
                        f"All current buffer entries are already in the existing buffer. Not merging."
                    )
                    merged_buffer = self.buffer
                save_size = merged_buffer.shape[0]
                torch.save(merged_buffer.cpu(), self.save_path + "/embed_buffer.pt")
            else:
                torch.save(self.buffer.cpu(), self.save_path + "/embed_buffer.pt")
            print(
                f"Saved embed buffer with {save_size} entries to {self.save_path}/embed_buffer.pt"
            )

    def load(self):
        if self.load_path is not None:
            if not os.path.exists(self.load_path + "/embed_buffer.pt"):
                raise ValueError(f"No embed buffer found at {self.load_path}")
            self.buffer = torch.load(self.load_path + "/embed_buffer.pt").to(
                next(self.embedder.parameters()).device
            )

    def reset(self):
        del self.buffer
        self.buffer = None
        self.load()

    def add(self, items: np.ndarray, embeddings=None):
        if self.buffer is None:
            self.buffer = self.embedder.embed(items)
        else:
            if embeddings is not None:
                new_embedding = embeddings
            else:
                new_embedding = self.embedder.embed(items)
            # check if new_embeddings is already in the buffer. and if it is, skip adding:
            new_embedding = self.get_unseen_elements(new_embedding)
            if new_embedding is None:
                return
            self.buffer = torch.cat([self.buffer, new_embedding], dim=0)
            if self.buffer.shape[0] > self.max_size:
                self.rationalize_buffer()

    def rationalize_buffer(self):
        print(
            f"Rationalizing buffer with current size {self.buffer.shape[0]} and max size {self.max_size}..."
        )
        # cluster down to half the size and keep the cluster centers only
        target_size = self.max_size // 2
        kmeans = KMeans(n_clusters=target_size, random_state=42)
        kmeans.fit(self.buffer.cpu().numpy())
        self.buffer = torch.tensor(
            kmeans.cluster_centers_,
            dtype=self.buffer.dtype,
            device=self.buffer.device,
        )

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        passed_frames = get_passed_frames(infos)
        with torch.no_grad():
            if self.buffer is None:
                self.add(passed_frames)
                return 0.0
            else:
                item_embeddings = self.embedder.embed(passed_frames)
                if self.similarity_metric == "cosine":
                    # assume they are normalized, so cosine similarity is just dot product
                    cosine_similarities = torch.matmul(
                        self.buffer, item_embeddings.T
                    ).T  # shape (n_frames, buffer_size)
                    # get max per frame, then average across frames
                    score = (
                        (1 - torch.max(cosine_similarities, dim=-1).values)
                        .mean()
                        .item()
                    )
                elif self.similarity_metric == "distance":
                    # compute pairwise distances and take min per frame, then average across frames
                    distances = torch.cdist(
                        item_embeddings, self.buffer
                    )  # shape (n_frames, buffer_size)
                    score = torch.min(distances, dim=-1).values.mean().item()
                elif self.similarity_metric == "hinge":
                    # essentially find the percentage of dimensions where item_embedding - self.buffer_element < margin, max over buffer elements, then average across frames
                    margin = 0.01
                    diffs = (
                        item_embeddings.unsqueeze(1) - self.buffer.unsqueeze(0)
                    ).abs()
                    hinge = (diffs < margin).float()
                    scores = hinge.mean(
                        dim=-1
                    )  # percentage of dimensions that are close
                    max_scores = torch.max(
                        scores, dim=-1
                    ).values  # max over buffer elements
                    score = (1 - max_scores).mean().item()  # average across frames
                self.add(passed_frames, embeddings=item_embeddings)
                return score


class ClusterOnlyBuffer:
    def __init__(self, embedder, load_path=None, save_path=None, n_clusters=100):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.save_path = save_path
        self.load_path = load_path
        if self.save_path is not None and self.save_path == self.load_path:
            print(
                f"Warning: save_path and load_path are the same. This means the buffer will be overwritten on reset and grow over time. This should only be used with a random agent to accumilate base observation data."
            )
        self.reset()

    def iterative_save(self):
        if self.save_path is not None:
            print(
                "ClusterOnlyBuffer does not support iterative saving. Clusters are only saved on reset to avoid excessive file I/O. Call save() method on reset instead."
            )
        return

    def load(self):
        if self.load_path is not None:
            if not os.path.exists(self.load_path + "/cluster_buffer.pkl"):
                raise ValueError(f"No cluster buffer found at {self.load_path}")
            with open(self.load_path + "/cluster_buffer.pkl", "rb") as f:
                self.clusters = pickle.load(f)
                self.has_fit = True
                self.initial_buffer = None

    def reset(self):
        self.clusters = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42)
        self.has_fit = False
        self.initial_buffer = None
        self.load()

    def add(self, items: np.ndarray):
        if self.has_fit:
            self.clusters.partial_fit(items)
        else:
            if self.initial_buffer is None:
                self.initial_buffer = items
            else:
                self.initial_buffer = np.concatenate(
                    [self.initial_buffer, items], axis=0
                )
                if len(self.initial_buffer) >= self.clusters.n_clusters:
                    self.clusters.fit(self.initial_buffer)
                    self.has_fit = True
                    self.initial_buffer = None

    def compare(self, items: np.ndarray) -> int:
        score = self.clusters.score(items)
        return -score

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        with torch.no_grad():
            passed_frames = get_passed_frames(infos)
            embedding = self.embedder.embed(passed_frames).cpu().numpy()
            if self.has_fit:
                score = self.compare(embedding)
            else:
                score = 0.0
            self.add(embedding)
            return score


def get_curiosity_module(args):
    if args.observation_embedder == "random_patch":
        embedder = PatchProjection(
            normalized_observations=args.similarity_metric == "cosine"
        ).eval()

    elif args.observation_embedder == "cnn":
        embedder = CNNEmbedder(
            normalized_observations=args.similarity_metric == "cosine"
        ).eval()
        if args.embedder_load_path is not None:
            embedder.load(args.embedder_load_path)
    if "buffer" in args.curiosity_module:
        if args.curiosity_module == "embedbuffer":
            module = EmbedBuffer(
                embedder,
                similarity_metric=args.similarity_metric,
                save_path=args.buffer_save_path,
                load_path=args.buffer_load_path,
            )
        elif args.curiosity_module == "clusterbuffer":
            module = ClusterOnlyBuffer(
                embedder=embedder,
                save_path=args.buffer_save_path,
                load_path=args.buffer_load_path,
            )
    elif args.curiosity_module == "world_model":
        module = WorldModel(embedder=embedder)
    return module


class PokemonReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        # self.screens = np.zeros(
        #    (self.buffer_size, self.n_envs, 144, 160),
        #    dtype=np.uint8,
        # )
        # screens not needed because its always the last element of the observations
        self.steps = -np.ones((self.buffer_size, self.n_envs), dtype=np.uint16)
        self.step_counts = np.zeros((self.n_envs,), dtype=np.uint16)

    def reset(self):
        # self.screens = np.zeros(
        #    (self.buffer_size, self.n_envs, 144, 160),
        #    dtype=np.uint8,
        # )
        self.steps = -np.ones((self.buffer_size, self.n_envs), dtype=np.uint16)
        self.step_counts = np.zeros((self.n_envs,), dtype=np.uint16)
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos,
    ):
        done = "final_info" in infos
        # self.screens[self.pos, 0] = get_passed_frames(infos)[-1].reshape(144, 160)
        self.steps[self.pos, :] = self.step_counts.copy()

        self.step_counts += 1
        self.step_counts = self.step_counts * (1 - done)  # reset step count on done
        super().add(obs, next_obs, action, reward, done, infos)

    def save(self, save_folder, run_name):
        print("Saving replay buffer...")
        if save_folder is not None:
            save_path = f"{save_folder}/{run_name}/"
            os.makedirs(save_path, exist_ok=True)
            save_size = None
            if self.full:
                np.save(save_path + "/observations.npy", self.observations)
                np.save(save_path + "/actions.npy", self.actions)
                np.save(save_path + "/rewards.npy", self.rewards)
                # np.save(save_path + "/screens.npy", self.screens)
                np.save(save_path + "/steps.npy", self.steps)
                save_size = self.buffer_size
                save_outliers(
                    self.observations,
                    self.actions,
                    self.rewards,
                    self.steps,
                    save_folder,
                    run_name,
                )
            else:
                np.save(save_path + "/observations.npy", self.observations[: self.pos])
                np.save(save_path + "/actions.npy", self.actions[: self.pos])
                np.save(save_path + "/rewards.npy", self.rewards[: self.pos])
                # np.save(save_path + "/screens.npy", self.screens[: self.pos])
                np.save(save_path + "/steps.npy", self.steps[: self.pos])
                save_size = self.pos
                save_outliers(
                    self.observations[: self.pos],
                    self.actions[: self.pos],
                    self.rewards[: self.pos],
                    self.steps[: self.pos],
                    save_folder,
                    run_name,
                )
            print(f"Saved replay buffer with {save_size} entries to {save_path}")


def stacked_frame_to_single(observation):
    # observation shape is (4, 144, 160)
    # We want to convert it to a (144 x 4, 160) image where each of the 4 frames is stacked vertically. This is just for visualization purposes.
    all_obs = observation.reshape(FRAME_STACK, 144, 160)
    show_obs = np.zeros((144 * FRAME_STACK, 160), dtype=np.uint8)
    for i in range(FRAME_STACK):
        show_obs[i * 144 : (i + 1) * 144] = all_obs[i]
    return show_obs


def plot_observation(
    observation, save_name, save_folder="../frame_saves/", title="Observation Frames"
):
    save_path = f"{save_folder}/{save_name}.png"
    os.makedirs(save_folder, exist_ok=True)
    obs_single = stacked_frame_to_single(observation)
    plt.figure(figsize=(5, 20))
    plt.imshow(obs_single, cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_transition(observation, new_observation, action, reward, step, save_path):
    obs_single = stacked_frame_to_single(observation)
    new_obs_single = stacked_frame_to_single(new_observation)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(obs_single, cmap="gray")
    action_class, action_kwargs = OneOfToDiscreteWrapper.get_high_level_action_static(
        action.reshape(-1)[0]
    )
    action = action_kwargs
    axes[0].set_title(f"\nStep {step.reshape(-1)[0]}\nObservation\nAction: {action}")
    axes[1].imshow(new_obs_single, cmap="gray")
    axes[1].set_title(f"New Observation\nReward: {reward.reshape(-1)[0]}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_outliers(
    observations,
    actions,
    rewards,
    steps,
    save_folder,
    run_name,
    n_samples=20,
    outlier_threshold=2,
):
    print("Analyzing rewards for outliers and visualization...")
    load_path = f"{save_folder}/{run_name}/"
    new_episode_indices = np.where(steps == 0)[0]
    last_step_indices = new_episode_indices - 1
    # replace -1 values with the last index of the buffer for the first episode
    minus_one_indices = np.where(last_step_indices == -1)[0]
    if len(minus_one_indices) > 0:
        last_step_indices[minus_one_indices] = len(steps) - 1
    np.save(load_path + "last_step_indices.npy", last_step_indices)
    reward_mean = np.nanmean(rewards)
    reward_std = np.nanstd(rewards)
    rewards[new_episode_indices] = reward_mean
    rewards[last_step_indices] = reward_mean
    reward_mean = rewards.mean()
    reward_std = rewards.std()
    normalized_rewards = (rewards - reward_mean) / (reward_std + 1e-8)
    # identify the indices of the top and bottom n_samples rewards
    sorted_indices = np.argsort(normalized_rewards, axis=0)
    top_sample_indices = sorted_indices[-n_samples:]
    top_sample_indices = top_sample_indices[::-1]
    bottom_sample_indices = sorted_indices[:n_samples]
    high_reward_indices = np.where(normalized_rewards > outlier_threshold)[0]
    if len(high_reward_indices) > 0:
        np.save(load_path + "high_reward_indices.npy", high_reward_indices)
        print(
            f"Saved {len(high_reward_indices)} reward indices to {load_path + 'high_reward_indices.npy'}"
        )
    else:
        print("No high reward outliers found.")

    save_path = f"{save_folder}/{run_name}/transition_visualizations/"
    os.makedirs(save_path, exist_ok=True)
    for i in range(n_samples):
        observation, new_observation, action, reward, step = (
            observations[top_sample_indices[i]],
            observations[top_sample_indices[i] + 1],
            actions[top_sample_indices[i]],
            rewards[top_sample_indices[i]],
            steps[top_sample_indices[i]],
        )
        visualize_transition(
            observation,
            new_observation,
            action,
            reward,
            step,
            save_path + f"top_transition_{i}.png",
        )
        observation, new_observation, action, reward, step = (
            observations[bottom_sample_indices[i]],
            observations[bottom_sample_indices[i] + 1],
            actions[bottom_sample_indices[i]],
            rewards[bottom_sample_indices[i]],
            steps[bottom_sample_indices[i]],
        )
        visualize_transition(
            observation,
            new_observation,
            action,
            reward,
            step,
            save_path + f"bottom_transition_{i}.png",
        )
    print(
        f"Saved transition visualizations for top and bottom {n_samples} rewards to {save_path}"
    )
