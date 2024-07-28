import os
import json
from stable_baselines3 import PPO
from diambra.arena import Arena, EnvironmentSettings, SpaceTypes

class MyAgent:
    def __init__(self):
        # Define all parameters directly in the code
        self.params = {
            "settings": {
                "game_id": "tektagt",
                "step_ratio": 6,
                "frame_shape": (128, 128, 1),
                "continue_game": 0.0,
                "action_space": "multi_discrete",
                "characters": ("Jin", "Heihachi"),
                "difficulty": 1,
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
            },
            "policy_kwargs": {
                "net_arch": [
                    {"pi": [512, 256, 128], "vf": [512, 256, 128]}
                ]
            },
            "ppo_settings": {
                "gamma": 0.92,
                "learning_rate": [0.00008, 0.000009],
                "clip_range": [0.1, 0.01],
                "batch_size": 64,
                "n_epochs": 15,
                "n_steps": 2048,
                "autosave_freq": 1000,
                "time_steps": 250000,
                "model_checkpoint": "1000000"
            }
        }

        # Define the paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_folder = os.path.join(base_dir, 'model')
        model_checkpoint = self.params["ppo_settings"]["model_checkpoint"]
        checkpoint_path = os.path.join(model_folder, f"{model_checkpoint}.zip")

        # Load the trained agent
        self.agent = PPO.load(checkpoint_path)

    def act(self, observation):
        action, _states = self.agent.predict(observation, deterministic=True)
        return action

def make_agent():
    return MyAgent()
