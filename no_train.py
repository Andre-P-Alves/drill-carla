import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
from PIL import Image
import wandb
import gym
from carla_gym.envs import LeaderboardEnv
from carla_gym.core.task_actor.scenario_actor.agents.constant_speed_agent import ConstantSpeedAgent
from carla_gym.utils.expert_noiser import ExpertNoiser
from rl_birdview_wrapper import RlBirdviewWrapper
from expert_dataset import ExpertDataset
from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs, obs_configs
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch_layers import XtMaCNN
import pandas as pd


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
    'carla_map': 'Town02',
    'weather_group': 'dynamic_1.0',
    'routes_group': 'train'
}

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
        },
        'gnss': {
            'module': 'navigation.gnss'
        },    
        'central_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,
            'location': [1.2, 0.0, 1.3],
            'rotation': [0.0, 0.0, 0.0]
        },
        'left_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,
            'location': [1.2, -0.25, 1.3],
            'rotation': [0.0, 0.0, -45.0]
        },
        'right_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,
            'location': [1.2, 0.25, 1.3],
            'rotation': [0.0, 0.0, 45.0]
        }
    }
}

observation_space = {}
observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8)
observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
observation_space = gym.spaces.Dict(**observation_space)
action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)
feature_extractor = XtMaCNN(observation_space, states_neurons=[256,256])

policy_kwargs = {
        'observation_space': observation_space,
        'action_space': action_space,
        'policy_head_arch': [256, 256],
        'feature_extractor': feature_extractor,
        'distribution_entry_point': 'distributions:BetaDistribution',
    }

device = 'cuda'

policy1 = AgentPolicy(**policy_kwargs)
policy1.to(device)

env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                         terminal_configs=terminal_configs, host="localhost", port=2000,
                         seed=2021, no_rendering=False, **env_configs)
env = RlBirdviewWrapper(env)

route_id = 0
ep_id = 0

expert_file_dir = Path('no_train')
expert_file_dir.mkdir(parents=True, exist_ok=True)
env.set_task_idx(route_id)
episode_dir = expert_file_dir / ('route_%02d' % route_id) / ('ep_%02d' % ep_id)
(episode_dir / 'birdview_masks').mkdir(parents=True, exist_ok=True)
ep_dict = {}
ep_dict['done'] = []
ep_dict['actions'] = []
ep_dict['state'] = []


def evaluate_policy(env, policy):
    obs = env.reset()

    done = False
    i_step = 0
    while i_step < 2000:
        obs_tensor_dict = dict([(k, th.as_tensor(v.copy()).unsqueeze(0).to(device)) for k, v in obs.items()])
        actions, _, _, _, _ = policy.forward(obs_tensor_dict, deterministic=True, clip_action=True)
        obs, _, done, _ = env.step(actions[0, :])
        ep_dict['actions'].append([actions[0, 0], actions[0, 1]])
        birdview = obs['birdview']
        for i_mask in range(1):
            birdview_mask = birdview[i_mask * 3: i_mask * 3 + 3]
            birdview_mask = np.transpose(birdview_mask, [1, 2, 0]).astype(np.uint8)
            Image.fromarray(birdview_mask).save(episode_dir / 'birdview_masks' / '{:0>4d}_{:0>2d}.png'.format(i_step, i_mask))
        ep_dict['state'].append(obs['state'])
        ep_dict['done'].append(done)
        i_step += 1
        if done:
           obs = env.reset()
    print(ep_dict)
    ep_df = pd.DataFrame(ep_dict)
    ep_df.to_json(episode_dir / 'episode.json')
evaluate_policy(env,policy1)