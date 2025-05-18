from diambra.arena import SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO
import diambra


if __name__ == "__main__":

    # Settings
    settings = EnvironmentSettings()
    settings.step_ratio = 6
    settings.difficulty = 1
    settings.outfits = 1
    settings.action_space = SpaceTypes.MULTI_DISCRETE
    settings.game_id = "tektagt"
    settings.characters = ("Jin", "Heihachi")

    # Wrappers Settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.frame_shape = (128, 128, 1)
    wrappers_settings.stack_frames = 6
    wrappers_settings.dilation = 1
    wrappers_settings.normalize_reward = True
    wrappers_settings.no_attack_buttons_combinations = True
    wrappers_settings.stack_actions = 12
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.flatten = True
    wrappers_settings.role_relative = True
    wrappers_settings.add_last_action = True
    wrappers_settings.filter_keys = ['action', 'frame', 'opp_active_character', 'opp_bar_status', 
        'opp_character', 'opp_character_1', 'opp_character_2', 'opp_health_1', 
        'opp_health_2', 'opp_side', 'own_active_character', 'own_bar_status', 
        'own_character', 'own_character_1', 'own_character_2', 'own_health_1', 
        'own_health_2', 'own_side', 'stage', 'timer'
    ]

    # Create environment
    env = diambra.arena.make("tektagt", settings, wrappers_settings)

    model_path = "./1000000"

    # Load agent
    agent = PPO.load(model_path, env)

    # Begin evaluation
    observation, info = env.reset()
    while True:
        action, _ = agent.predict(observation, deterministic=False)
        observation, reward, terminated, truncated, info = env.step(action.tolist())

        if terminated or truncated:
            observation, info = env.reset()
            if info["env_done"] is True:
                break

    env.close()

