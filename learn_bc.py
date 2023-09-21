import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
import wandb
import gym

from randon_dataset import ExpertDataset
from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs, obs_configs
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch_layers import XtMaCNN

env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}


def learn_bc(polcies, device, eval_loader):
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_dir / 'checkpoint.txt'

    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(parents=True, exist_ok=True)


    run = wandb.init(project='gail-carla2', reinit=True)
    with open(last_checkpoint_path, 'w') as log_file:
        log_file.write(wandb.run.path)
    start_ep = 0
    i_steps = 0

    episodes = 350
    ent_weight = 0.01
    min_eval_loss = np.inf
    for i_episode in tqdm.tqdm(range(start_ep, episodes)):
        for policy_i, policy_data in enumerate(policies):
            policy = policy_data['policy']
            train_loader = policy_data['train_loader']
            optimizer = policy_data['optimizer']
            total_loss = 0
            i_batch = 0
            policy = policy.train()
            # Expert dataset
            for expert_batch in train_loader:
                expert_obs_dict, expert_action = expert_batch
                obs_tensor_dict = {
                    'state': expert_obs_dict['state'].float().to(device),
                    'birdview': expert_obs_dict['birdview'].float().to(device)
                }
                expert_action = expert_action.to(device)

                # Get BC loss
                alogprobs, entropy_loss = policy.evaluate_actions(obs_tensor_dict, expert_action)
                bcloss = -alogprobs.mean()

                loss = bcloss + ent_weight * entropy_loss
                total_loss += loss
                i_batch += 1
                if policy_i == 0:
                    i_steps += expert_obs_dict['state'].shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_eval_loss = 0
            i_eval_batch = 0
            for expert_batch in eval_loader:
                expert_obs_dict, expert_action = expert_batch
                obs_tensor_dict = {
                    'state': expert_obs_dict['state'].float().to(device),
                    'birdview': expert_obs_dict['birdview'].float().to(device)
                }
                expert_action = expert_action.to(device)

                # Get BC loss
                with th.no_grad():
                    alogprobs, entropy_loss = policy.evaluate_actions(obs_tensor_dict, expert_action)
                bcloss = -alogprobs.mean()

                eval_loss = bcloss + ent_weight * entropy_loss
                total_eval_loss += eval_loss
                i_eval_batch += 1
        
            loss = total_loss / i_batch
            eval_loss = total_eval_loss / i_eval_batch
            wandb.log({
                f'loss_{policy_i}': loss,
                f'eval_loss_{policy_i}': eval_loss,
            }, step=i_steps)
            ckpt_path = (ckpt_dir / f'bc3_ckpt_{i_episode}_min_policy_{policy_i}_eval.pth').as_posix()
            th.save(
                {'policy_state_dict': policy.state_dict()},
               ckpt_path
            )
        # if min_eval_loss > eval_loss:
        #     ckpt_path = (ckpt_dir / f'bc_ckpt_{i_episode}_min_eval.pth').as_posix()
        #     th.save(
        #         {'policy_state_dict': policy.state_dict()},
        #        ckpt_path
        #     )
        #     min_eval_loss = eval_loss

        # train_init_kwargs = {
        #     'start_ep': i_episode,
        #     'i_steps': i_steps
        # } 
        # ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        # th.save({'policy_state_dict': policy.state_dict(),
        #          'train_init_kwargs': train_init_kwargs},
        #         ckpt_path)
        # wandb.save(f'./{ckpt_path}')
    run = run.finish()


if __name__ == '__main__':
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

    device = 'cuda'
    policies = []
    batch_size = 24
    n_emsamble = 5
    for i in range(n_emsamble):
        policy = AgentPolicy(**policy_kwargs)
        policy.to(device)

        gail_train_loader = th.utils.data.DataLoader(
            ExpertDataset(
                'gail_experts',
                n_routes=3,
                n_eps=1,
                subsample_frequency=1#n_emsamble
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = optim.Adam(policy.parameters(), lr=1e-5)

        
        policies.append({'policy': policy, 'train_loader': gail_train_loader, 'optimizer': optimizer})

    
    gail_val_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=1,
            n_eps=1,
            route_start=4
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    learn_bc(policies, device, gail_val_loader)