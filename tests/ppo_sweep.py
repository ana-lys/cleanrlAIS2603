import itertools
import subprocess

# ------------------------------------------------------------
# 1. Hyperparameter grid
# ------------------------------------------------------------
param_grid = {
    "tag": ["v2"],
    "seed": [4],
    "anneal_lr": [False],
    "clip_vloss": [False],
    "ent_coef": [0.001],
    "update_epochs": [8],
    "learning_rate": [5e-4],
    "env_id": ["LunarLander-v3"],
    "total_timesteps": [15000000],
}
# param_grid = {
#     "tag": ["v2"],
#     "seed": [4],
#     "anneal_lr": [False],
#     "clip_vloss": [False, True],
#     "ent_coef": [0.01 , 0.001],
#     "update_epochs": [4 , 8],
#     "learning_rate": [5e-4 , 1e-4],
#     "env_id": ["LunarLander-v3"],
#     "total_timesteps": [7500000],
# }

# ------------------------------------------------------------
# 2. Path to your PPO script
# ------------------------------------------------------------
BASE_CMD = ["python", "cleanrl/ppo.py"]

# ------------------------------------------------------------
# 3. Optional settings
# ------------------------------------------------------------
DRY_RUN = False          # Set True to only print commands
MAX_RUNS = None          # Limit number of runs for testing

# ------------------------------------------------------------
# 4. Generate all combinations
# ------------------------------------------------------------
keys = list(param_grid.keys())
values = list(param_grid.values())

combinations = list(itertools.product(*values))
total_combos = len(combinations)
print(f"Total hyperparameter combinations: {total_combos}")

if MAX_RUNS is not None:
    combinations = combinations[:MAX_RUNS]
    print(f"Limiting to first {MAX_RUNS} runs.")

# ------------------------------------------------------------
# 5. Build and launch each command
# ------------------------------------------------------------
for i, combo in enumerate(combinations, 1):
    params = dict(zip(keys, combo))

    cmd = BASE_CMD.copy()
    for k, v in params.items():
        # Convert underscore to hyphen for command line
        flag = k.replace('_', '-')
        if isinstance(v, bool):
            # Boolean flags: use --flag for True, --no-flag for False
            if v:
                cmd.append(f"--{flag}")
            else:
                cmd.append(f"--no-{flag}")
        else:
            # Non‑boolean: --flag value
            cmd.append(f"--{flag}")
            cmd.append(str(v))

    print(f"\n[{i}/{len(combinations)}] Running: {' '.join(cmd)}")

    if not DRY_RUN:
        subprocess.run(cmd)
    else:
        print("DRY RUN: command printed, not executed.")

print("\nSweep finished.")