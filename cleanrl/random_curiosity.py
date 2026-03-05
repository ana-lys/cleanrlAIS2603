# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
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
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """ Just for API """

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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.exp_name = depathify(args.exp_name)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    curiosity_module = get_curiosity_module(args)
    start_time = time.time()
    episode_rewards = []

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        actions = np.array(
            [envs.single_action_space.sample() for _ in range(envs.num_envs)]
        )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            this_episode_reward = sum(episode_rewards)
            episode_rewards = []
            if args.reset_curiosity_module:
                curiosity_module.reset()  # reset the curiosity module at the end of each episode if the flag is set
        else:
            rewards[0] = rewards[0] + curiosity_module.get_reward(
                obs, actions, next_obs, infos
            )
            episode_rewards.append(rewards[0])

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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
                    writer.add_scalar(
                        "charts/completion_percentage",
                        int(100 * global_step / args.total_timesteps),
                        global_step,
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % 100 == 0:
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS",
                int(global_step / (time.time() - start_time)),
                global_step,
            )

    rb.save(args.replay_buffer_save_folder, args.exp_name)
    envs.close()
    writer.close()

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
