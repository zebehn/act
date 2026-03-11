from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation as R

from constants import DT
from device_utils import dataloader_pin_memory, resolve_device
from visualize_episodes import save_videos

TARGET_ACT_DIM = 14
LIBERO_CAMERA_NAMES = ['agentview_rgb', 'eye_in_hand_rgb']
LIBERO_OBS_IMAGE_KEYS = {
    'agentview_rgb': 'agentview_image',
    'eye_in_hand_rgb': 'robot0_eye_in_hand_image',
}
LIBERO_TASK_CONFIGS = {
    'libero_object_task0': {
        'dataset_type': 'libero_demo_hdf5',
        'libero_suite': 'libero_object',
        'libero_task_id': 0,
        'dataset_rel_path': 'libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5',
        'camera_names': list(LIBERO_CAMERA_NAMES),
        'state_dim': TARGET_ACT_DIM,
        'action_dim': TARGET_ACT_DIM,
        'source_state_dim': 9,
        'source_action_dim': 7,
        'eval_num_rollouts': 50,
        'state_source': 'joint_gripper',
    }
}


def _read_libero_config_yaml() -> Dict[str, str]:
    config_path = os.environ.get('LIBERO_CONFIG_PATH', os.path.expanduser('~/.libero'))
    yaml_path = os.path.join(config_path, 'config.yaml')
    if not os.path.exists(yaml_path):
        return {}
    try:
        import yaml
    except Exception:
        return {}
    with open(yaml_path, 'r') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return dict(data or {})


def get_libero_dataset_root() -> str:
    for env_key in ('LIBERO_DATASET_ROOT', 'LIBERO_DATA_DIR', 'LIBERO_ROOT'):
        value = os.environ.get(env_key)
        if not value:
            continue
        candidate = os.path.expanduser(value)
        if os.path.basename(candidate) == 'datasets':
            return candidate
        datasets_candidate = os.path.join(candidate, 'datasets')
        if os.path.isdir(datasets_candidate):
            return datasets_candidate
        return candidate
    config = _read_libero_config_yaml()
    if 'datasets' in config:
        return os.path.expanduser(config['datasets'])
    return os.path.expanduser('~/LIBERO/datasets')


def get_libero_bddl_root() -> Optional[str]:
    config = _read_libero_config_yaml()
    return os.path.expanduser(config['bddl_files']) if 'bddl_files' in config else None


def get_libero_init_states_root() -> Optional[str]:
    config = _read_libero_config_yaml()
    return os.path.expanduser(config['init_states']) if 'init_states' in config else None


def resolve_libero_task_config(task_name: str) -> Dict[str, Any]:
    if task_name not in LIBERO_TASK_CONFIGS:
        raise KeyError(f'Unknown LIBERO task {task_name}')
    cfg = deepcopy(LIBERO_TASK_CONFIGS[task_name])
    dataset_root = get_libero_dataset_root()
    dataset_path = os.path.join(dataset_root, cfg['dataset_rel_path'])
    cfg['dataset_dir'] = dataset_path
    cfg['dataset_path'] = dataset_path
    cfg['episode_len'] = cfg.get('episode_len', infer_libero_episode_len(dataset_path, fallback=400))
    cfg['num_episodes'] = cfg.get('num_episodes', infer_libero_num_episodes(dataset_path, fallback=50))
    cfg['state_source'] = os.environ.get('LIBERO_STATE_SOURCE', cfg.get('state_source', 'joint_gripper'))
    eval_override = os.environ.get('LIBERO_EVAL_NUM_ROLLOUTS')
    if eval_override:
        cfg['eval_num_rollouts'] = int(eval_override)
    return cfg


def _sorted_demo_names(root: h5py.File) -> List[str]:
    demos = sorted(list(root['data'].keys()), key=lambda name: int(name.split('_')[-1]))
    return demos


def infer_libero_num_episodes(dataset_path: str, fallback: int = 50) -> int:
    if not os.path.exists(dataset_path):
        return fallback
    with h5py.File(dataset_path, 'r') as root:
        return len(_sorted_demo_names(root))


def infer_libero_episode_len(dataset_path: str, fallback: int = 400) -> int:
    if not os.path.exists(dataset_path):
        return fallback
    with h5py.File(dataset_path, 'r') as root:
        demos = _sorted_demo_names(root)
        if not demos:
            return fallback
        return max(int(root[f'data/{demo}/actions'].shape[0]) for demo in demos)


def list_libero_episode_ids(dataset_path: str) -> List[int]:
    if not os.path.exists(dataset_path):
        return []
    with h5py.File(dataset_path, 'r') as root:
        return list(range(len(_sorted_demo_names(root))))


def load_libero_demo_names(dataset_path: str) -> List[str]:
    with h5py.File(dataset_path, 'r') as root:
        return _sorted_demo_names(root)




def _libero_demo_qpos(obs_group) -> np.ndarray:
    joint_states = obs_group['joint_states'][()].astype(np.float32)
    gripper_states = obs_group['gripper_states'][()].astype(np.float32)
    return np.concatenate([joint_states, gripper_states], axis=-1)


def _libero_demo_ee_qpos(obs_group) -> np.ndarray:
    ee_pos = obs_group['ee_pos'][()].astype(np.float32)
    ee_ori = obs_group['ee_ori'][()].astype(np.float32)
    gripper_states = obs_group['gripper_states'][()].astype(np.float32)
    return np.concatenate([ee_pos, ee_ori, gripper_states], axis=-1)

def pad_to_act_dim(array: np.ndarray, target_dim: int = TARGET_ACT_DIM) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.shape[-1] > target_dim:
        raise ValueError(f'Cannot pad array with dim {array.shape[-1]} into target_dim {target_dim}')
    out_shape = array.shape[:-1] + (target_dim,)
    padded = np.zeros(out_shape, dtype=np.float32)
    padded[..., :array.shape[-1]] = array
    return padded


class LiberoEpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_path, camera_names, norm_stats, max_seq_len, demo_names=None, state_source='joint_gripper'):
        super().__init__()
        self.episode_ids = list(episode_ids)
        self.dataset_path = dataset_path
        self.camera_names = list(camera_names)
        self.norm_stats = norm_stats
        self.max_seq_len = int(max_seq_len)
        self.is_sim = True
        self.demo_names = demo_names or load_libero_demo_names(dataset_path)
        self.state_source = state_source
        if not self.episode_ids:
            raise ValueError('episode_ids must be non-empty')

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        demo_name = self.demo_names[self.episode_ids[index]]
        with h5py.File(self.dataset_path, 'r') as root:
            obs_group = root[f'data/{demo_name}/obs']
            actions = root[f'data/{demo_name}/actions'][()].astype(np.float32)
            if self.state_source == 'ee_gripper':
                qpos_seq = _libero_demo_ee_qpos(obs_group)
            else:
                qpos_seq = _libero_demo_qpos(obs_group)
            episode_len = int(actions.shape[0])
            start_ts = np.random.choice(episode_len)
            qpos = pad_to_act_dim(qpos_seq[start_ts])
            image_dict = {cam: obs_group[cam][start_ts] for cam in self.camera_names}
            action = pad_to_act_dim(actions[start_ts:])
            action_len = episode_len - start_ts

        padded_action = np.zeros((self.max_seq_len, TARGET_ACT_DIM), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_seq_len, dtype=np.float32)
        is_pad[action_len:] = 1.0
        all_cam_images = np.stack([image_dict[cam] for cam in self.camera_names], axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data) / 255.0
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        action_data = (action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        qpos_data = (qpos_data - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']
        return image_data, qpos_data, action_data, is_pad


def get_libero_norm_stats(dataset_path: str, episode_ids: Sequence[int], demo_names: Optional[Sequence[str]] = None):
    return get_libero_norm_stats_for_state_source(dataset_path, episode_ids, demo_names=demo_names, state_source='joint_gripper')


def get_libero_norm_stats_for_state_source(dataset_path: str, episode_ids: Sequence[int], demo_names: Optional[Sequence[str]] = None, state_source: str = 'joint_gripper'):
    demo_names = list(demo_names or load_libero_demo_names(dataset_path))
    all_qpos_data = []
    all_action_data = []
    example_qpos = None
    with h5py.File(dataset_path, 'r') as root:
        for episode_idx in episode_ids:
            demo_name = demo_names[episode_idx]
            obs_group = root[f'data/{demo_name}/obs']
            if state_source == 'ee_gripper':
                qpos = pad_to_act_dim(_libero_demo_ee_qpos(obs_group))
            else:
                qpos = pad_to_act_dim(_libero_demo_qpos(obs_group))
            action = pad_to_act_dim(root[f'data/{demo_name}/actions'][()].astype(np.float32))
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
            if example_qpos is None and len(qpos) > 0:
                example_qpos = qpos[0]
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = torch.clip(all_action_data.std(dim=[0], keepdim=True), 1e-2, np.inf)
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = torch.clip(all_qpos_data.std(dim=[0], keepdim=True), 1e-2, np.inf)
    return {
        'action_mean': action_mean.numpy().squeeze(),
        'action_std': action_std.numpy().squeeze(),
        'qpos_mean': qpos_mean.numpy().squeeze(),
        'qpos_std': qpos_std.numpy().squeeze(),
        'example_qpos': example_qpos,
    }


def load_libero_data(task_config: Dict[str, Any], batch_size_train: int, batch_size_val: int, device=None, episode_ids=None):
    dataset_path = task_config['dataset_path']
    camera_names = task_config['camera_names']
    if episode_ids is None:
        episode_ids = list_libero_episode_ids(dataset_path)
    else:
        episode_ids = list(episode_ids)
    if not episode_ids:
        raise ValueError(f'No LIBERO demos found in {dataset_path}')
    demo_names = load_libero_demo_names(dataset_path)
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(episode_ids)
    split_idx = int(train_ratio * len(episode_ids))
    train_indices = shuffled_indices[:split_idx].tolist() or shuffled_indices[:1].tolist()
    val_indices = shuffled_indices[split_idx:].tolist() or train_indices[:1]
    state_source = task_config.get('state_source', 'joint_gripper')
    norm_stats = get_libero_norm_stats_for_state_source(dataset_path, episode_ids, demo_names=demo_names, state_source=state_source)
    if os.environ.get('LIBERO_DISABLE_ACTION_NORM') == '1':
        norm_stats['action_mean'] = np.zeros_like(norm_stats['action_mean'])
        norm_stats['action_std'] = np.ones_like(norm_stats['action_std'])
    max_seq_len = task_config['episode_len']
    train_dataset = LiberoEpisodicDataset(train_indices, dataset_path, camera_names, norm_stats, max_seq_len, demo_names=demo_names, state_source=state_source)
    val_dataset = LiberoEpisodicDataset(val_indices, dataset_path, camera_names, norm_stats, max_seq_len, demo_names=demo_names, state_source=state_source)
    pin_memory = dataloader_pin_memory(device) if device is not None else False
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=pin_memory, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, pin_memory=pin_memory, num_workers=1)
    return train_loader, val_loader, norm_stats, train_dataset.is_sim


def _import_libero_modules():
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        return benchmark, OffScreenRenderEnv
    except ModuleNotFoundError:
        import sys
        repo_path = os.environ.get('LIBERO_REPO_PATH')
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        return benchmark, OffScreenRenderEnv


def make_libero_env(task_config: Dict[str, Any]):
    benchmark, OffScreenRenderEnv = _import_libero_modules()
    suite_name = task_config['libero_suite']
    task_id = int(task_config['libero_task_id'])
    benchmark_instance = benchmark.get_benchmark_dict()[suite_name]()
    task = benchmark_instance.get_task(task_id)
    bddl_root = get_libero_bddl_root()
    if not bddl_root:
        raise RuntimeError('LIBERO bddl path could not be resolved. Set LIBERO_CONFIG_PATH or LIBERO_DATA_DIR.')
    env_args = {
        'bddl_file_name': os.path.join(bddl_root, task.problem_folder, task.bddl_file),
        'camera_heights': 128,
        'camera_widths': 128,
        'camera_names': ['agentview', 'robot0_eye_in_hand'],
    }
    dataset_path = task_config.get('dataset_path')
    if dataset_path and os.path.exists(dataset_path):
        with h5py.File(dataset_path, 'r') as root:
            env_args_json = root['data'].attrs.get('env_args')
            if env_args_json:
                env_meta = json.loads(env_args_json)
                env_kwargs = env_meta.get('env_kwargs', {})
                controller_type = None
                if 'controller_configs' in env_kwargs and isinstance(env_kwargs['controller_configs'], dict):
                    controller_type = env_kwargs['controller_configs'].get('type')
                for key, value in env_kwargs.items():
                    if key in ('bddl_file_name', 'controller_configs'):
                        continue
                    env_args[key] = value
                if controller_type:
                    env_args['controller'] = controller_type
                env_args['bddl_file_name'] = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(**env_args)
    init_states = benchmark_instance.get_task_init_states(task_id)
    return env, init_states, task, benchmark_instance


def libero_obs_to_qpos(obs) -> np.ndarray:
    return libero_obs_to_qpos_with_source(obs, state_source='joint_gripper')


def libero_obs_to_qpos_with_source(obs, state_source: str = 'joint_gripper') -> np.ndarray:
    if state_source == 'ee_gripper':
        eef_pos = np.asarray(obs['robot0_eef_pos'], dtype=np.float32)
        eef_quat = np.asarray(obs['robot0_eef_quat'], dtype=np.float32)
        ee_rotvec = R.from_quat(eef_quat).as_rotvec().astype(np.float32)
        return pad_to_act_dim(np.concatenate([eef_pos, ee_rotvec, np.asarray(obs['robot0_gripper_qpos'], dtype=np.float32)], axis=0))
    if 'robot_states' in obs:
        return pad_to_act_dim(obs['robot_states'])
    if 'robot0_joint_pos' in obs and 'robot0_gripper_qpos' in obs:
        return pad_to_act_dim(np.concatenate([obs['robot0_joint_pos'], obs['robot0_gripper_qpos']], axis=0))
    return pad_to_act_dim(np.concatenate([obs['robot0_gripper_qpos'], obs['robot0_eef_pos'], obs['robot0_eef_quat']], axis=0))


def libero_obs_to_image(obs, camera_names: Sequence[str], device) -> torch.Tensor:
    curr_images = []
    for cam_name in camera_names:
        obs_key = LIBERO_OBS_IMAGE_KEYS.get(cam_name, cam_name)
        curr_image = rearrange(obs[obs_key], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    return torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)


def select_libero_init_state(init_states, seed: int):
    idx = int(seed) % len(init_states)
    return np.asarray(init_states[idx]).copy(), idx


def eval_libero_bc(config: Dict[str, Any], ckpt_name: str, save_episode: bool = True):
    from imitate_episodes import make_policy
    set_seed = __import__('utils').set_seed
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    device = config['device']
    task_config = config['task_config']
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(loading_status)
    policy.to(device)
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = json.loads(json.dumps({})) if False else __import__('pickle').load(f)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    env, init_states, _task, _benchmark = make_libero_env(task_config)
    num_rollouts = int(task_config.get('eval_num_rollouts', 50))
    episode_returns = []
    highest_rewards = []
    state_dim = config['state_dim']
    query_frequency = policy_config['num_queries']
    temporal_agg = config['temporal_agg']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    with torch.inference_mode():
        for rollout_id in range(num_rollouts):
            env.seed(1000 + rollout_id)
            env.reset()
            init_state = np.asarray(init_states[rollout_id % len(init_states)]).copy()
            obs = env.set_init_state(init_state)
            if temporal_agg:
                all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim], device=device)
                all_time_action_mask = torch.zeros([max_timesteps, max_timesteps + num_queries], dtype=torch.bool, device=device)
            image_list = []
            rewards = []
            for t in range(max_timesteps):
                image_list.append({cam: obs[LIBERO_OBS_IMAGE_KEYS.get(cam, cam)] for cam in camera_names})
                qpos_numpy = libero_obs_to_qpos_with_source(obs, task_config.get('state_source', 'joint_gripper'))
                qpos = torch.from_numpy(pre_process(qpos_numpy)).float().to(device).unsqueeze(0)
                curr_image = libero_obs_to_image(obs, camera_names, device)
                if config['policy_class'] == 'ACT':
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        all_time_action_mask[[t], t:t + num_queries] = True
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = all_time_action_mask[:, t]
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).float().to(device).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % policy_config['num_queries']]
                else:
                    raw_action = policy(qpos, curr_image)
                env_action = raw_action.squeeze(0).detach().cpu().numpy()[:task_config['source_action_dim']]
                obs, reward, done, info = env.step(env_action)
                rewards.append(float(reward if reward is not None else 0.0))
                if done:
                    break
            rewards = np.array(rewards, dtype=np.float32)
            episode_return = float(np.sum(rewards))
            episode_returns.append(episode_return)
            highest_reward = float(np.max(rewards)) if len(rewards) else 0.0
            highest_rewards.append(highest_reward)
            print(f'Rollout {rollout_id}\n{episode_return=}, {highest_reward=}, env_max_reward=1.0, Success: {highest_reward >= 1.0}')
            if save_episode:
                save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
    success_rate = np.mean(np.array(highest_rewards) >= 1.0)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(2):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'
    print(summary_str)
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))
    env.close()
    return success_rate, avg_return
