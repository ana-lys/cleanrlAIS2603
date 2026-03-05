# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rainbow/#rainbow_ataripy
import collections
import copy
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.port_poke_worlds import (
    depathify,
    get_curiosity_module,
    get_gameboy_cnn_chain,
    PokemonReplayBuffer as ReplayBuffer,
    save_all_models,
    MaxLengthList,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    model_save_path: str | None = None
    """custom path to save the model (overrides default `runs/{run_name}/{exp_name}.cleanrl_model`)"""
    model_save_ranks: int | None = 3
    """ will save the final model as well as the `model_save_ranks` top models during training according to episodic return. Only applicable if `save_model` is True."""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0000625
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 8000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    n_step: int = 3
    """the number of steps to look ahead for n-step Q learning"""
    prioritized_replay_alpha: float = 0.5
    """alpha parameter for prioritized replay buffer"""
    prioritized_replay_beta: float = 0.4
    """beta parameter for prioritized replay buffer"""
    prioritized_replay_eps: float = 1e-6
    """epsilon parameter for prioritized replay buffer"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""

    # Curiosity module specific arguments
    curiosity_module: str = "embedbuffer"
    """the type of curiosity module to use."""
    observation_embedder: str = "random_patch"
    """the type of observation embedder to use for the curiosity module."""
    embedder_load_path: str | None = None
    """path to load the observation embedder's weights from. Only applicable if the observation embedder supports loading."""
    reset_curiosity_module: bool = True
    """whether to reset the curiosity module at the end of each episode"""
    similarity_metric: str = "cosine"
    """the similarity metric to use for the EmbedBuffer curiosity module."""
    buffer_save_path: str | None = None
    """ path to save the curiosity module's buffer """
    buffer_load_path: str | None = None
    """ path to load the curiosity module's buffer from """
    replay_buffer_save_folder: str | None = None
    """ folder to save the replay buffer """


def make_env(env_id, seed, idx, capture_video, run_name, gamma=0.99):
    if env_id.startswith("poke_worlds"):
        from cleanrl_utils.port_poke_worlds import poke_worlds_make_env

        return poke_worlds_make_env(
            env_id, seed, idx, capture_video, run_name, gamma=gamma
        )

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        # factorized gaussian noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


# ALGO LOGIC: initialize agent here:
class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = env.single_action_space.n
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.network = nn.Sequential(*get_gameboy_cnn_chain())
        obs_shape = env.single_observation_space.shape
        with torch.inference_mode():
            output_dim = self.network(torch.zeros(1, *obs_shape)).shape[1]

        conv_output_size = output_dim

        self.value_head = nn.Sequential(
            NoisyLinear(conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms)
        )

        self.advantage_head = nn.Sequential(
            NoisyLinear(conv_output_size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_atoms * self.n_actions),
        )

    def forward(self, x):
        h = self.network(x / 255.0)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist

    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


PrioritizedBatch = collections.namedtuple(
    "PrioritizedBatch",
    [
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "dones",
        "indices",
        "weights",
    ],
)


# adapted from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class SumSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[parent * 2 + 1] + self.tree[parent * 2 + 2]
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def total(self):
        return self.tree[0]

    def retrieve(self, value):
        idx = 0
        while idx * 2 + 1 < self.tree_size:
            left = idx * 2 + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)


# adapted from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class MinSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.full(self.tree_size, float("inf"), dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = min(
                self.tree[parent * 2 + 1], self.tree[parent * 2 + 2]
            )
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def min(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(
        self, capacity, obs_shape, device, n_step, gamma, alpha=0.6, beta=0.4, eps=1e-6
    ):
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.buffer_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.buffer_next_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.buffer_actions = np.zeros(capacity, dtype=np.int64)
        self.buffer_rewards = np.zeros(capacity, dtype=np.float32)
        self.buffer_dones = np.zeros(capacity, dtype=np.bool_)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)

        # For n-step returns
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        reward = 0.0
        next_obs = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        for i in range(len(self.n_step_buffer)):
            reward += self.gamma**i * self.n_step_buffer[i][2]
            if self.n_step_buffer[i][4]:
                next_obs = self.n_step_buffer[i][3]
                done = True
                break
        return reward, next_obs, done

    def add(self, obs, action, reward, next_obs, done):
        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_obs, done = self._get_n_step_info()
        obs = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]

        idx = self.pos
        self.buffer_obs[idx] = obs
        self.buffer_next_obs[idx] = next_obs
        self.buffer_actions[idx] = action[0]
        self.buffer_rewards[idx] = reward[0]
        self.buffer_dones[idx] = done

        priority = self.max_priority**self.alpha
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size):
        indices = []
        p_total = self.sum_tree.total()
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        samples = {
            "observations": torch.from_numpy(self.buffer_obs[indices]).to(self.device),
            "actions": torch.from_numpy(self.buffer_actions[indices])
            .to(self.device)
            .unsqueeze(1),
            "rewards": torch.from_numpy(self.buffer_rewards[indices])
            .to(self.device)
            .unsqueeze(1),
            "next_observations": torch.from_numpy(self.buffer_next_obs[indices]).to(
                self.device
            ),
            "dones": torch.from_numpy(self.buffer_dones[indices])
            .to(self.device)
            .unsqueeze(1),
        }

        probs = np.array(
            [self.sum_tree.tree[idx + self.capacity - 1] for idx in indices]
        )
        weights = (self.size * probs / p_total) ** -self.beta
        weights = weights / weights.max()
        samples["weights"] = torch.from_numpy(weights).to(self.device).unsqueeze(1)
        samples["indices"] = indices

        return PrioritizedBatch(**samples)

    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())

        for idx, priority in zip(indices, priorities):
            priority = priority**self.alpha
            self.sum_tree.update(idx, priority)
            self.min_tree.update(idx, priority)


def get_model_save_data(args, q_network):
    model_data = {
        "model_weights": copy.deepcopy(q_network.state_dict()),
        "args": vars(args),
    }
    return model_data


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.exp_name = depathify(args.exp_name)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    assert (
        args.buffer_save_path is None or args.buffer_save_path != args.buffer_load_path
    ), "buffer save path and load path cannot be the same for this algorithm."
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    q_network = NoisyDuelingDistributionalNetwork(
        envs, args.n_atoms, args.v_min, args.v_max
    ).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=1.5e-4)
    target_network = NoisyDuelingDistributionalNetwork(
        envs, args.n_atoms, args.v_min, args.v_max
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = PrioritizedReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
        args.n_step,
        args.gamma,
        args.prioritized_replay_alpha,
        args.prioritized_replay_beta,
        args.prioritized_replay_eps,
    )
    curiosity_module = get_curiosity_module(args)
    model_data_list = MaxLengthList(args.model_save_ranks) if args.save_model else None
    model_reward_list = (
        MaxLengthList(args.model_save_ranks) if args.save_model else None
    )

    start_time = time.time()
    episode_rewards = []

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # anneal PER beta to 1
        rb.beta = min(
            1.0,
            args.prioritized_replay_beta
            + global_step * (1.0 - args.prioritized_replay_beta) / args.total_timesteps,
        )

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            q_dist = q_network(torch.Tensor(obs).to(device))
            q_values = torch.sum(q_dist * q_network.support, dim=2)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            this_episode_reward = sum(episode_rewards)
            if args.save_model:
                insert_index = model_reward_list.do_item_insert(this_episode_reward)
                if insert_index is not None:
                    model_data = get_model_save_data(args, q_network)
                    model_data_list.insert_item(insert_index, model_data)
            episode_rewards = []
            if args.reset_curiosity_module:
                curiosity_module.reset()  # reset the curiosity module at the end of each episode if the flag is set
        else:
            rewards[0] = rewards[0] + curiosity_module.get_reward(
                obs, actions, next_obs, infos
            )
            episode_rewards.append(rewards[0])

        if "final_info" in infos:
            if isinstance(infos["final_info"], dict):
                infos["final_info"] = [infos["final_info"]]
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={this_episode_reward}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", this_episode_reward, global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]
        rb.add(obs, actions, rewards, real_next_obs, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                # reset the noise for both networks
                q_network.reset_noise()
                target_network.reset_noise()
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    next_dist = target_network(
                        data.next_observations
                    )  # [B, num_actions, n_atoms]
                    support = target_network.support  # [n_atoms]
                    next_q_values = torch.sum(
                        next_dist * support, dim=2
                    )  # [B, num_actions]

                    # double q-learning
                    next_dist_online = q_network(
                        data.next_observations
                    )  # [B, num_actions, n_atoms]
                    next_q_online = torch.sum(
                        next_dist_online * support, dim=2
                    )  # [B, num_actions]
                    best_actions = torch.argmax(next_q_online, dim=1)  # [B]
                    next_pmfs = next_dist[
                        torch.arange(args.batch_size), best_actions
                    ]  # [B, n_atoms]

                    # compute the n-step Bellman update.
                    gamma_n = args.gamma**args.n_step
                    next_atoms = data.rewards + gamma_n * support * (
                        1 - data.dones.float()
                    )
                    tz = next_atoms.clamp(q_network.v_min, q_network.v_max)

                    # projection
                    delta_z = q_network.delta_z
                    b = (tz - q_network.v_min) / delta_z  # shape: [B, n_atoms]
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)

                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (
                        u.float() + (l == b).float() - b
                    ) * next_pmfs  # [B, n_atoms]
                    d_m_u = (b - l) * next_pmfs  # [B, n_atoms]

                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                dist = q_network(data.observations)  # [B, num_actions, n_atoms]
                pred_dist = dist.gather(
                    1, data.actions.unsqueeze(-1).expand(-1, -1, args.n_atoms)
                ).squeeze(1)
                log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))

                loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)
                loss = (loss_per_sample * data.weights.squeeze()).mean()

                # update priorities
                new_priorities = loss_per_sample.detach().cpu().numpy()
                rb.update_priorities(data.indices, new_priorities)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), global_step)
                    q_values = (pred_dist * q_network.support).sum(dim=1)  # [B]
                    writer.add_scalar(
                        "losses/q_values", q_values.mean().item(), global_step
                    )
                    sps = int(global_step / (time.time() - start_time))
                    print("SPS:", sps)
                    writer.add_scalar("charts/SPS", sps, global_step)
                    writer.add_scalar("charts/beta", rb.beta, global_step)
                    writer.add_scalar(
                        "charts/completion_percentage",
                        global_step / args.total_timesteps,
                        global_step,
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_param, param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1.0 - args.tau) * target_param.data
                    )

    rb.save(args.replay_buffer_save_folder, args.exp_name)
    envs.close()
    writer.close()

    if args.save_model:
        final_model_data = get_model_save_data(args, q_network)
        save_all_models(
            final_model_data=final_model_data,
            model_data_list=model_data_list,
            model_save_folder=args.model_save_path,
        )

    if args.capture_video:
        video_candidates = [
            f for f in os.listdir(f"videos/{run_name}") if f.endswith(".mp4")
        ]
        # is in format rl-video-episode-episode_id.mp4
        # sort by episode_id
        video_candidates.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        for video in video_candidates:
            episode_id = int(video.split("-")[-1].split(".")[0])
            wandb.log(
                {
                    f"video/{episode_id}": wandb.Video(
                        f"videos/{run_name}/{video}", format="mp4"
                    )
                }
            )
