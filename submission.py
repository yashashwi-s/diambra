#!/usr/bin/env python3
import os
import yaml
import json
import argparse
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO

# Custom YAML constructor for Python tuples
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

def main(cfg_file, model_path=None):
    # Read the cfg file
    with open(cfg_file, 'r') as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.Loader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")
    tensor_board_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                        params["folders"]["model_name"], "tb")

    os.makedirs(model_folder, exist_ok=True)

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])

    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings)
    print("Activated {} environment(s)".format(num_envs))

    # Load the trained agent
    if model_path is not None:
        agent = PPO.load(model_path, env=env)
        print("Loaded trained agent from", model_path)
    else:
        raise ValueError("Model path must be provided to evaluate the trained model.")

    # Evaluate the agent
    print("\nStarting trained agent evaluation ...\n")
    observation = env.reset()
    total_reward = 0
    for _ in range(10000):  # Run for a number of steps
        env.render()
        action, _state = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Action: {action}, Reward: {reward}, Info: {info}")
        if done:
            print(f"Total Reward: {total_reward}")
            observation = env.reset()
            break
    print("\n... trained agent evaluation completed.\n")
    print("Total reward:", total_reward)

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--modelPath", type=str, required=True, help="Path to the trained model")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile, opt.modelPath)
