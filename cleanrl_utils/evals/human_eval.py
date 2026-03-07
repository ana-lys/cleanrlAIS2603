import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from cleanrl_utils.port_gameboy_worlds import (
    plot_observation,
    OneOfToDiscreteWrapper,
)


def evaluate(
    model_path: None,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: None,
    device: torch.device = None,
    capture_video: bool = True,
    args=None,
):
    eval_episodes = 1
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, 0, capture_video, run_name)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    obs, _ = envs.reset()
    episodic_returns = []
    curiosity_rewards = []
    input_sequence = Model
    n_steps = 0
    while len(episodic_returns) < eval_episodes:
        action = input_sequence[n_steps]
        n_steps += 1
        actions = np.array([action for _ in range(envs.num_envs)])
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        curiosity_reward = args.curiosity_module.get_reward(
            obs, actions, next_obs, infos
        )
        title = f"Step: {n_steps}, Action: {OneOfToDiscreteWrapper.get_high_level_action_static(action)}, Curiosity Reward: {curiosity_reward:6f}"
        save_name = f"eval_{n_steps}.png"
        plot_observation(next_obs[0], save_name=save_name, title=title)
        print(
            f"Step: {n_steps}, Action: {OneOfToDiscreteWrapper.get_high_level_action_static(action)}, Curiosity Reward: {curiosity_reward:6f}"
        )
        rewards[0] = rewards[0] + curiosity_reward
        curiosity_rewards.append(curiosity_reward)
        if "final_info" in infos:
            if isinstance(infos["final_info"], dict):
                infos["final_info"] = [infos["final_info"]]
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}, curiosity_reward={sum(curiosity_rewards):6f}"
                )
                episodic_returns += [info["episode"]["r"]]
            args.curiosity_module.iterative_save()
            args.curiosity_module.reset()
            curiosity_rewards = []
        elif n_steps >= len(input_sequence):
            print(
                f"input_sequence exhausted, resetting env. curiosity_reward={sum(curiosity_rewards):6f}"
            )
            episodic_returns += [0]
            args.curiosity_module.iterative_save()
            args.curiosity_module.reset()
            curiosity_rewards = []
            obs, _ = envs.reset()
            n_steps = 0
            continue
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.dqn import QNetwork, make_env

    model_path = hf_hub_download(
        repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="q_network.pth"
    )
    evaluate(
        model_path,
        make_env,
        "CartPole-v1",
        eval_episodes=10,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False,
    )
