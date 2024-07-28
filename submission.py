import os
import yaml
import numpy as np
from stable_baselines3 import PPO
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena import SpaceTypes

# Load configuration
with open("config.yaml", "r") as yaml_file:
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)

# Settings
params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])
wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])

# Create environment
env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings)
print(f"Activated {num_envs} environment(s)")

# Load the trained agent
model_path = os.path.join("model", "your_model.zip")
agent = PPO.load(model_path)

def main():
    obs = env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

if __name__ == "__main__":
    main()
