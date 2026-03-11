import torch
import numpy as np
import os
import pickle
import argparse
import glob
import re
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from device_utils import resolve_device
from task_registry import is_libero_task, resolve_task_config, list_task_episode_ids
from libero_adapter import eval_libero_bc

import IPython
e = IPython.embed


def list_available_episode_ids(dataset_dir):
    pattern = os.path.join(dataset_dir, 'episode_*.hdf5')
    episode_ids = []
    for path in glob.glob(pattern):
        match = re.search(r'episode_(\d+)\.hdf5$', os.path.basename(path))
        if match:
            episode_ids.append(int(match.group(1)))
    return sorted(set(episode_ids))


def resolve_resume_checkpoint(resume_ckpt, ckpt_dir, device):
    if not resume_ckpt:
        return None

    candidate = None
    if resume_ckpt == 'auto':
        epoch_ckpts = glob.glob(os.path.join(ckpt_dir, 'policy_epoch_*_seed_*.ckpt'))
        epoch_ckpts = sorted(epoch_ckpts, key=lambda p: int(re.search(r'policy_epoch_(\d+)_seed_', os.path.basename(p)).group(1)) if re.search(r'policy_epoch_(\d+)_seed_', os.path.basename(p)) else -1, reverse=True)
        candidates = epoch_ckpts + [os.path.join(ckpt_dir, 'policy_best.ckpt'), os.path.join(ckpt_dir, 'policy_last.ckpt')]
    else:
        candidates = [resume_ckpt]

    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            torch.load(path, map_location=device)
            candidate = path
            break
        except Exception as e:
            print(f'Skipping invalid checkpoint {path}: {e}')

    return candidate


def main(args):
    set_seed(args['seed'])
    device = resolve_device(args.get('device', 'auto'))
    print(f'Using device: {device}')
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    task_config = resolve_task_config(task_name)
    is_libero = is_libero_task(task_name)
    is_sim = task_config.get('task_source') == 'sim'
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    available_episode_ids = list_task_episode_ids(task_config)
    if len(available_episode_ids) == 0:
        raise FileNotFoundError(f'No episodes found for task {task_name} at {dataset_dir}')
    if len(available_episode_ids) < num_episodes:
        print(
            f'Found {len(available_episode_ids)} local episodes in {dataset_dir}; '
            f'overriding requested num_episodes={num_episodes}.'
        )
        num_episodes = len(available_episode_ids)
    selected_episode_ids = available_episode_ids[:num_episodes]

    # fixed parameters
    state_dim = task_config.get('state_dim', 14)
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'device': device,
        'resume_ckpt': args.get('resume_ckpt'),
        'task_config': task_config,
    }

    if is_eval:
        ckpt_names = args.get('eval_ckpt_names') or ['policy_best.ckpt']
        save_episode = not args.get('no_save_episode', False)
        results = []
        for ckpt_name in ckpt_names:
            if is_libero:
                success_rate, avg_return = eval_libero_bc(config, ckpt_name, save_episode=save_episode)
            else:
                success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=save_episode)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        device=device,
        episode_ids=selected_episode_ids,
        task_config=task_config,
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, device):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    device = config['device']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(loading_status)
    policy.to(device)
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    box_pose_ref = None
    if 'sim_transfer_cube' in task_name or 'sim_insertion' in task_name:
        from sim_env import BOX_POSE as box_pose_ref
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            box_pose_ref[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            box_pose_ref[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim], device=device)
            all_time_action_mask = torch.zeros([max_timesteps, max_timesteps+num_queries], dtype=torch.bool, device=device)

        qpos_history = torch.zeros((1, max_timesteps, state_dim), device=device)
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names, device)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        all_time_action_mask[[t], t:t+num_queries] = True
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = all_time_action_mask[:, t]
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).float().to(device).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy, device):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device, non_blocking=True)
    qpos_data = qpos_data.to(device, non_blocking=True)
    action_data = action_data.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    device = config['device']
    resume_ckpt = config.get('resume_ckpt')

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)

    start_epoch = 0
    resume_path = resolve_resume_checkpoint(resume_ckpt, ckpt_dir, device)
    if resume_path is not None:
        policy.load_state_dict(torch.load(resume_path, map_location=device))
        match = re.search(r'policy_epoch_(\d+)_seed_', os.path.basename(resume_path))
        if match:
            start_epoch = int(match.group(1)) + 1
        print(f'Resumed model weights from {resume_path}, start_epoch={start_epoch}')

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    if start_epoch >= num_epochs:
        print(f'start_epoch ({start_epoch}) >= num_epochs ({num_epochs}); skipping training loop.')
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for _, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, device)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary['loss'].item()
            best_ckpt_info = (max(start_epoch - 1, 0), epoch_val_loss, deepcopy(policy.state_dict()))

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, device)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        epoch_start_idx = len(train_history)
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[epoch_start_idx:])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    if best_ckpt_info is None:
        if len(validation_history) > 0:
            fallback_val = validation_history[-1]['loss'].item()
        else:
            fallback_val = float('inf')
        best_ckpt_info = (max(start_epoch - 1, 0), fallback_val, deepcopy(policy.state_dict()))

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    if len(train_history) > 0 and len(validation_history) > 0:
        plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    else:
        print('Skipping train/val plot generation: insufficient history.')

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    if len(train_history) == 0 or len(validation_history) == 0:
        return
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--eval_ckpt_names', nargs='+', default=['policy_best.ckpt'], help='Checkpoint filename(s) inside ckpt_dir to evaluate when --eval is set')
    parser.add_argument('--no_save_episode', action='store_true', help='Disable rollout video saving during --eval')
    parser.add_argument('--device', action='store', type=str, default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--resume_ckpt', action='store', type=str, default=None, help='Checkpoint path or "auto"')
    
    main(vars(parser.parse_args()))
