import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
import wandb
import gym
import pandas as pd
from expert_dataset import ExpertDataset
from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs, obs_configs
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch_layers import XtMaCNN

observation_space = {}
observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8)
observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
observation_space = gym.spaces.Dict(**observation_space)

action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)
feature_extractor = XtMaCNN(observation_space, states_neurons=[256,256])
    # network

policy_kwargs = {
        'observation_space': observation_space,
        'action_space': action_space,
        'policy_head_arch': [256, 256],
        'feature_extractor': feature_extractor,
        'distribution_entry_point': 'distributions:BetaDistribution',
    }
policies = []
i_episode = 749
n_ensemble = 5
device = 'cuda'
ckpt_dir = Path('ckpt')
for policy_i in range(n_ensemble):
    policy = AgentPolicy(**policy_kwargs)
    policy.to(device)
    ckpt_path = (ckpt_dir / f'bc3_ckpt_{i_episode}_min_policy_{policy_i}_eval.pth').as_posix()
    saved_variables = th.load(ckpt_path, map_location='cuda')
    policy.load_state_dict(saved_variables['policy_state_dict'])
    policies.append(policy)

batch_size = 1

eval_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=1,
            n_eps=1,
            route_start=4
        ),
        batch_size=batch_size,
        shuffle=True,
    )
no_train_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'no_train',
            n_routes=1,
            n_eps=1,
            route_start=0
        ),
        batch_size=batch_size,
        shuffle=True,
    )

def calc_vars(samples_loader,fds):
    q = 0.98
    actions = []
    variances = []
    quantiles = []
    quantiles_2 = []
    cov_1 = []
    cov_2 = []
    for expert_batch in samples_loader:
        ensemble_actions = []
        for policy in policies:
            expert_obs_dict, expert_action = expert_batch
            obs_tensor_dict = {
                        'state': expert_obs_dict['state'].float().to(device),
                        'birdview': expert_obs_dict['birdview'].float().to(device)
                    }
            expert_action = expert_action.to(device)

                # Get BC loss
            with th.no_grad():
                
                action, log_probs, mu, sigma, _ = policy.forward(obs_tensor_dict, deterministic=True, clip_action=True)
                ensemble_actions.append(action)
                
        teste = np.vstack(ensemble_actions)
        ensemble_actions = np.transpose(np.array(ensemble_actions), (1, 0, 2))
        actions.extend(ensemble_actions.tolist())
        cov = np.cov(teste.T)
        #print(cov, cov[0,0], cov[1,1])
        var = np.matmul(np.matmul(expert_action.cpu(), cov), expert_action.cpu().T).item()
        variances.append(var)
        quantiles.append(np.quantile(np.array(var), q))
        cov_1.append(cov[0,0])
        cov_2.append(cov[1,1])

    ensemble_action = np.array(actions)
    acc_action = ensemble_action[:, :, 0]
    steer_action = ensemble_action[:, :, 1]
    #print(acc_action.shape)
    acc_var = acc_action.var(1)
    steer_var = steer_action.var(1)
    df = pd.DataFrame({'acc_action': acc_action.tolist(), 'steer_action': steer_action.tolist()})
    df.to_json(f'actions_var{fds}.json')
    df = pd.DataFrame({'cov_1': cov_1, 'cov_2': cov_2})
    df.to_json(f'covariances{fds}.json')
    quantilzada = np.quantile(np.array(var), q)
    print(quantilzada)
    return acc_var, steer_var, variances, quantiles

acc_var, steer_var, variancias, quantis = calc_vars(eval_loader, 1)
df = pd.DataFrame({'acc_var': acc_var, 'steer_var': steer_var})
df.to_json('eval_var.json')
df = pd.DataFrame({'variance': variancias, 'quantile': quantis})
df.to_json('eval_varquan.json')

acc_var, steer_var, variancias, quantis = calc_vars(no_train_loader, 2)
df = pd.DataFrame({'acc_var': acc_var, 'steer_var': steer_var})
df.to_json('no_train_var.json')
df = pd.DataFrame({'variance': variancias, 'quantile': quantis})
df.to_json('no_train_varquan.json')