#!/usr/bin/env python3
import os
import json
from stable_baselines3 import PPO
from diambra.arena import SpaceTypes, load_settings_flat_dict, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env

def main():
    # Define the configuration settings directly in the script
    params = {
        "folders": {
            "parent_dir": "./results/",
            "model_name": "newer"
        },
        "settings": {
            "game_id": "tektagt",
            "step_ratio": 6,
            "frame_shape": (128, 128, 1),
            "continue_game": 0.0,
            "action_space": "multi_discrete",
            "characters": ("Jin", "Heihachi"),
            "difficulty": 7,
            "outfits": 1
        },
        "wrappers_settings": {
            "normalize_reward": True,
            "no_attack_buttons_combinations": True,
            "stack_frames": 6,
            "dilation": 1,
            "add_last_action": True,
            "stack_actions": 12,
            "scale": True,
            "exclude_image_scaling": True,
            "role_relative": True,
            "flatten": True,
            "filter_keys": [
                'action', 'frame', 'opp_active_character', 'opp_bar_status', 'opp_character',
                'opp_character_1', 'opp_character_2', 'opp_health_1', 'opp_health_2', 'opp_side',
                'own_active_character', 'own_bar_status', 'own_character', 'own_character_1',
                'own_character_2', 'own_health_1', 'own_health_2', 'own_side', 'stage', 'timer'
            ]
        }
    }

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "model", "1.zip")

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])

    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings)
    print("Activated {} environment(s)".format(num_envs))

    # Load the trained agent
    if os.path.exists(model_path):
        agent = PPO.load(model_path, env=env)
        print("Loaded trained agent from", model_path)
    else:
        raise ValueError("Model path must be provided to evaluate the trained model.")

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Evaluate the agent
    print("\nStarting trained agent evaluation ...\n")
    try:
        observation = env.reset()
    except Exception as e:
        print(f"Error during environment reset: {e}")
        return 1

    total_reward = 0
    for _ in range(10000):  # Run for a number of steps
        try:
            action, _state = agent.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                observation = env.reset()
                break
        except Exception as e:
            print(f"Error during environment step: {e}")
            break

    print("\n... trained agent evaluation completed.\n")
    print("Total reward:", total_reward)

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    main()
