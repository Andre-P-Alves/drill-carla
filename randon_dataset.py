# modified from https://github.com/rohitrango/BC-regularized-GAIL/blob/master/a2c_ppo_acktr/algo/gail.py

from pathlib import Path
from typing import Union, Dict, Tuple, Any
from functools import partial
import time
import torch as th
import numpy as np
import pandas as pd

from PIL import Image


class ExpertDataset(th.utils.data.Dataset):
    def __init__(self, dataset_directory, n_routes=1, n_eps=1, route_start=0, ep_start=0, subsample_frequency=1):
        self.dataset_path = Path(dataset_directory)
        self.length = 0
        self.get_idx = []
        self.trajs_states = []
        self.trajs_actions = []

        for route_idx in range(route_start, route_start + n_routes):
            for ep_idx in range(ep_start, ep_start + n_eps):
                route_path = self.dataset_path / ('route_%02d' % route_idx) / ('ep_%02d' % ep_idx)
                route_df = pd.read_json(route_path / 'episode.json')
                traj_length = route_df.shape[0]
                for _ in range(int(traj_length/subsample_frequency)):
                    step_idx = np.random.randint(0, traj_length)
                    self.get_idx.append((route_idx, ep_idx, step_idx))
                    state_dict = {}
                    for state_key in route_df.columns:
                        state_dict[state_key] = route_df.iloc[step_idx][state_key]
                    self.trajs_states.append(state_dict)
                    self.trajs_actions.append(th.Tensor(route_df.iloc[step_idx]['actions']))
                    self.length += 1

        self.trajs_actions = th.stack(self.trajs_actions)
        self.actual_obs = [None for _ in range(self.length)]

    def __len__(self):
        return self.length

    def process_image(self, image_path):
        image_array = Image.open(image_path)
        image_array = np.transpose(image_array, [2, 0, 1])
        image_tensor = th.as_tensor(image_array.copy())
        return image_tensor

    def __getitem__(self, j):
        route_idx, ep_idx, step_idx = self.get_idx[j]
        if self.actual_obs[j] is None:
            # Load only the first time, images in uint8 are supposed to be light
            ep_dir = self.dataset_path / 'route_{:0>2d}/ep_{:0>2d}'.format(route_idx, ep_idx)
            masks_list = []
            for mask_index in range(1):
                mask_tensor = self.process_image(ep_dir / 'birdview_masks/{:0>4d}_{:0>2d}.png'.format(step_idx, mask_index))
                masks_list.append(mask_tensor)
            birdview = th.cat(masks_list)

            obs_dict = {
                'birdview': birdview,
                'item_idx': j
            }

            state_dict = self.trajs_states[j]
            for state_key in state_dict:
                obs_dict[state_key] = th.Tensor(state_dict[state_key])
            self.actual_obs[j] = obs_dict
        else:
            obs_dict = self.actual_obs[j]

        return obs_dict, self.trajs_actions[j]
