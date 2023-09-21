# modified from https://github.com/zhejz/carla-roach/blob/main/train_rl.py

from pathlib import Path
import wandb
import torch as th

from carla_gym.envs import EndlessEnv, LeaderboardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList

from rl_birdview_wrapper import RlBirdviewWrapper
from ppo import PPO
from ppo_policy import PpoPolicy
from wandb_callback import WandbCallback

RESUME_LAST_TRAIN = False

reward_configs = {
    'hero': {
        'entry_point': 'reward.valeo_action:ValeoAction',
        'kwargs': {}
    }
}

terminal_configs = {
    'hero': {
        'entry_point': 'terminal.valeo_no_det_px:ValeoNoDetPx',
        'kwargs': {}
    }
}

env_configs = {
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}

env_eval_configs = {
    'weather_group': 'simple',
    'routes_group': 'train'
}

multi_env_configs = [
    {"host": "localhost", "port": 2000, 'carla_map': 'Town01'},
    {"host": "localhost", "port": 2002, 'carla_map': 'Town01'},
    {"host": "localhost", "port": 2004, 'carla_map': 'Town01'},
    {"host": "localhost", "port": 2006, 'carla_map': 'Town01'},
]

multi_env_eval_configs = [
    {"host": "localhost", "port": 2008, 'carla_map': 'Town02'},
]

def get_obs_configs(rgb=False):
    obs_configs = {
        'hero': {
            'speed': {
                'module': 'actor_state.speed'
            },
            'control': {
                'module': 'actor_state.control'
            },
            'velocity': {
                'module': 'actor_state.velocity'
            },
            'birdview': {
                'module': 'birdview.chauffeurnet',
                'width_in_pixels': 192,
                'pixels_ev_to_bottom': 40,
                'pixels_per_meter': 5.0,
                'history_idx': [-16, -11, -6, -1],
                'scale_bbox': True,
                'scale_mask_col': 1.0
            },
            'route_plan': {
                'module': 'navigation.waypoint_plan',
                'steps': 20
            }
        }
    }
    return obs_configs



def env_maker(env_id, config, obs_configs, rendering=True):
    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs,
                    seed=2021, no_rendering=(not rendering), **env_configs, **config)
    env = RlBirdviewWrapper(env)
    return env


def env_eval_maker(env_id, config, obs_configs, rendering=True):
    env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs,
                    seed=2021, no_rendering=(not rendering), **env_eval_configs, **config)
    env = RlBirdviewWrapper(env)
    return env


if __name__ == '__main__':
    obs_configs = get_obs_configs()

    env = SubprocVecEnv([lambda env_id=env_id, config=config: env_maker(env_id, config, obs_configs, rendering=False) for env_id, config in enumerate(multi_env_configs)])
    env_eval = SubprocVecEnv([lambda env_id=env_id, config=config: env_eval_maker(env_id, config, obs_configs, rendering=False) for env_id, config in enumerate(multi_env_eval_configs)])
    features_extractor_entry_point = 'torch_layers:XtMaCNN'


    policy_kwargs = {
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'policy_head_arch': [256, 256],
        'value_head_arch': [256, 256],
        'features_extractor_entry_point': features_extractor_entry_point,
        'features_extractor_kwargs': {'states_neurons': [256,256]},
        'distribution_entry_point': 'distributions:BetaDistribution'
    }
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_dir / 'checkpoint.txt'
    wandb_run_id = None

    train_kwargs = {
        'initial_learning_rate': 2e-5,
        'n_steps_total': 3072,
        'batch_size': 256,
        'n_epochs': 20,
        'gamma': 0.99,
        'gae_lambda': 0.9,
        'clip_range': 0.2,
        'clip_range_vf': 0.2,
        'ent_coef': 0.01,
        'explore_coef': 0.05,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'lr_decay': 0.96,
        'use_exponential_lr_decay': True,
        'update_adv': False,
    }

    if RESUME_LAST_TRAIN:
        with open(last_checkpoint_path, 'r') as f:
            wb_run_path = f.read()
        api = wandb.Api()
        wandb_run = api.run(wb_run_path)
        wandb_run_id = wandb_run.id
        all_ckpts = [ckpt_file for ckpt_file in wandb_run.files() if 'ckpt_latest' in ckpt_file.name]
        ckpt_file = all_ckpts[0]
        ckpt_file.download(replace=True)
        ckpt_file_path = ckpt_file.name
        saved_variables = th.load(ckpt_file_path, map_location='cuda')
        train_kwargs = saved_variables['train_init_kwargs']

        policy = PpoPolicy(**saved_variables['policy_init_kwargs'])
        policy.load_state_dict(saved_variables['policy_state_dict'])

    else:
        policy = PpoPolicy(**policy_kwargs)

    agent = PPO(
        policy=policy,
        env=env,
        **train_kwargs
    )

    wb_callback = WandbCallback(env, env_eval, wandb_run_id)
    callback = CallbackList([wb_callback])
    with open(last_checkpoint_path, 'w') as log_file:
        log_file.write(wandb.run.path)
    agent.learn(total_timesteps=1e8, callback=callback)