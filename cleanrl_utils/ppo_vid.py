# eval_lunar.py (minimal version)
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# Paste your exact Agent class from ppo.py here
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential( ... )   # copy-paste your networks
        self.actor  = nn.Sequential( ... )

    def get_action_and_value(self, x, action=None):
        # copy-paste your method
        ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "./runs/your_run_name/agent.pt"   # ← your wandb-synced file

# Dummy to init spaces (or hardcode)
dummy_env = gym.make("LunarLander-v3")
agent = Agent([dummy_env]).to(device)               # pass fake vector env or just spaces
agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
agent.eval()

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos_lunar_eval",
                  episode_trigger=lambda ep: True,     # record every episode
                  name_prefix="ppo-lunar-eval")

for episode in range(10):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # add batch dim
    done = False
    ep_reward = 0

    while not done:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
            action = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = terminated or truncated

    print(f"Episode {episode+1}: Return = {ep_reward:.2f}")

env.close()
print("Videos saved in videos_lunar_eval/")