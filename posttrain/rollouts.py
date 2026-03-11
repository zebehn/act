from __future__ import annotations

import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from einops import rearrange
from torch.distributions import Normal

from constants import DT
from device_utils import resolve_device
from posttrain.common import build_act_policy, get_task_config, load_policy_checkpoint, save_json
from posttrain.schema import ROLLOUT_SCHEMA_VERSION, save_rollout_record
from sim_env import BOX_POSE, make_sim_env
from libero_adapter import libero_obs_to_image, libero_obs_to_qpos, make_libero_env, select_libero_init_state
from task_registry import is_libero_task


@dataclass
class RolloutStats:
    qpos_mean: np.ndarray
    qpos_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def from_pickle(cls, path: str) -> 'RolloutStats':
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        return cls(
            qpos_mean=np.asarray(stats['qpos_mean'], dtype=np.float32),
            qpos_std=np.asarray(stats['qpos_std'], dtype=np.float32),
            action_mean=np.asarray(stats['action_mean'], dtype=np.float32),
            action_std=np.asarray(stats['action_std'], dtype=np.float32),
        )

    def normalize_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return (qpos - self.qpos_mean) / self.qpos_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std


def sample_initial_object_pose(task_name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if 'sim_transfer_cube' in task_name:
        x = rng.uniform(0.0, 0.2)
        y = rng.uniform(0.4, 0.6)
        z = 0.05
        quat = np.array([1, 0, 0, 0], dtype=np.float32)
        return np.concatenate([np.array([x, y, z], dtype=np.float32), quat]).astype(np.float32)
    if 'sim_insertion' in task_name:
        peg = np.concatenate([
            np.array([
                rng.uniform(0.1, 0.2),
                rng.uniform(0.4, 0.6),
                0.05,
            ], dtype=np.float32),
            np.array([1, 0, 0, 0], dtype=np.float32),
        ])
        socket = np.concatenate([
            np.array([
                rng.uniform(-0.2, -0.1),
                rng.uniform(0.4, 0.6),
                0.05,
            ], dtype=np.float32),
            np.array([1, 0, 0, 0], dtype=np.float32),
        ])
        return np.concatenate([peg, socket]).astype(np.float32)
    raise NotImplementedError(f'Unsupported task for rollout sampling: {task_name}')


def set_task_initial_pose(task_name: str, pose: np.ndarray):
    if 'sim_transfer_cube' in task_name or 'sim_insertion' in task_name:
        BOX_POSE[0] = np.asarray(pose, dtype=np.float32)
        return
    raise NotImplementedError(f'Unsupported task for rollout reset pose: {task_name}')


def _make_task_env(task_name: str, task_config: Dict[str, Any]):
    if is_libero_task(task_name):
        env, init_states, _task, _benchmark = make_libero_env(task_config)
        return env, init_states, 1.0
    env = make_sim_env(task_name)
    return env, None, float(env.task.max_reward)


def _select_task_initial_state(task_name: str, init_state_pool, seed: int):
    if is_libero_task(task_name):
        init_state, init_state_id = select_libero_init_state(init_state_pool, seed)
        return np.asarray(init_state, dtype=np.float32), {'init_state_id': int(init_state_id)}
    pose = sample_initial_object_pose(task_name, seed)
    return np.asarray(pose, dtype=np.float32), {}


def _reset_task_env(env, task_name: str, initial_state: np.ndarray):
    if is_libero_task(task_name):
        env.reset()
        return env.set_init_state(initial_state)
    set_task_initial_pose(task_name, initial_state)
    return env.reset()


def _extract_obs(task_name: str, state_obj):
    if is_libero_task(task_name):
        return state_obj
    return state_obj.observation


def _extract_qpos(task_name: str, obs) -> np.ndarray:
    if is_libero_task(task_name):
        return libero_obs_to_qpos(obs)
    return np.asarray(obs['qpos'], dtype=np.float32)


def _step_task_env(env, task_name: str, action: np.ndarray, source_action_dim: Optional[int] = None):
    if is_libero_task(task_name):
        env_action = np.asarray(action[: source_action_dim or 7], dtype=np.float32)
        obs, reward, done, _info = env.step(env_action)
        return obs, float(reward if reward is not None else 0.0), bool(done)
    ts = env.step(action)
    reward = float(ts.reward if ts.reward is not None else 0.0)
    return ts, reward, False


def get_image_from_obs(task_name: str, obs, camera_names: Sequence[str], device: torch.device) -> torch.Tensor:
    if is_libero_task(task_name):
        return libero_obs_to_image(obs, camera_names, device)
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    return torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)


def get_image_from_ts(ts, camera_names: Sequence[str], device: torch.device) -> torch.Tensor:
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    return torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)


def _make_normal(mean: torch.Tensor, std: torch.Tensor) -> Normal:
    return Normal(mean, std.clamp_min(1e-6))


def _aggregate_normals(means: List[torch.Tensor], stds: List[torch.Tensor], decay_k: float = 0.01) -> Normal:
    if len(means) == 1:
        return _make_normal(means[0], stds[0])
    weights = torch.exp(-decay_k * torch.arange(len(means), device=means[0].device, dtype=means[0].dtype))
    weights = weights / weights.sum()
    stacked_means = torch.stack(means, dim=0)
    stacked_stds = torch.stack(stds, dim=0)
    agg_mean = (stacked_means * weights.view(-1, 1, 1)).sum(dim=0)
    agg_var = ((stacked_stds ** 2) * (weights.view(-1, 1, 1) ** 2)).sum(dim=0)
    return _make_normal(agg_mean, agg_var.sqrt())


def _resolve_rollout_sampling_strategy(
    sampling_strategy: str,
    deterministic: bool,
    checkpoint_load_status,
) -> str:
    if deterministic:
        return 'deterministic'
    if sampling_strategy != 'auto':
        return sampling_strategy
    missing_keys = set(getattr(checkpoint_load_status, 'missing_keys', []))
    legacy_missing = {'model.action_log_std', 'model.value_head.weight', 'model.value_head.bias'}
    if missing_keys & legacy_missing:
        return 'mean_gaussian'
    return 'model'


def _sample_rollout_action(dist: Normal, strategy: str, exploration_std: float, force_deterministic: bool = False) -> torch.Tensor:
    if force_deterministic or strategy == 'deterministic':
        return dist.mean
    if strategy == 'model':
        return dist.sample()
    if strategy == 'mean_gaussian':
        noise = torch.randn_like(dist.mean) * exploration_std
        return dist.mean + noise
    raise ValueError(f'Unsupported rollout sampling strategy: {strategy}')


class ChunkDistributionHistory:
    def __init__(self, num_queries: int, temporal_agg: bool = True, decay_k: float = 0.01, detach_history: bool = True):
        self.num_queries = num_queries
        self.temporal_agg = temporal_agg
        self.decay_k = decay_k
        self.detach_history = detach_history
        self.history: List[Dict[str, torch.Tensor]] = []

    def step_distribution(self, policy, qpos: torch.Tensor, image: torch.Tensor, t: int) -> Tuple[Normal, torch.Tensor]:
        if self.temporal_agg or not self.history or t % self.num_queries == 0:
            policy_out = policy.get_action_distribution(qpos, image)
            mean = policy_out['mean']
            std = policy_out['dist'].stddev
            value = policy_out['state_value']
            record = {
                'mean': mean.detach() if self.detach_history else mean,
                'std': std.detach() if self.detach_history else std,
                'value': value.detach() if self.detach_history else value,
            }
            self.history.append(record)
        current_value = self.history[-1]['value']
        if not self.temporal_agg:
            query_idx = t % self.num_queries
            dist = _make_normal(self.history[-1]['mean'][:, query_idx], self.history[-1]['std'][:, query_idx])
            return dist, current_value

        component_means: List[torch.Tensor] = []
        component_stds: List[torch.Tensor] = []
        for tau, record in enumerate(self.history):
            query_idx = t - tau
            if 0 <= query_idx < self.num_queries:
                component_means.append(record['mean'][:, query_idx])
                component_stds.append(record['std'][:, query_idx])
        if not component_means:
            raise RuntimeError(f'No temporal-agg components available for timestep {t}')
        return _aggregate_normals(component_means, component_stds, decay_k=self.decay_k), current_value


class TrajectoryReplay:
    def __init__(self, task_name: str, camera_names: Sequence[str], initial_object_pose: np.ndarray, actions_env: np.ndarray, task_config: Optional[Dict[str, Any]] = None):
        self.task_name = task_name
        self.camera_names = list(camera_names)
        self.initial_object_pose = np.asarray(initial_object_pose, dtype=np.float32)
        self.actions_env = np.asarray(actions_env, dtype=np.float32)
        self.task_config = task_config or get_task_config(task_name)

    def iter_observations(self):
        env, _init_state_pool, _env_max_reward = _make_task_env(self.task_name, self.task_config)
        state_obj = _reset_task_env(env, self.task_name, self.initial_object_pose)
        try:
            for action in self.actions_env:
                obs = _extract_obs(self.task_name, state_obj)
                yield obs
                state_obj, _reward, done = _step_task_env(
                    env,
                    self.task_name,
                    action,
                    source_action_dim=self.task_config.get('source_action_dim'),
                )
                if done:
                    break
        finally:
            close = getattr(env, 'close', None)
            if callable(close):
                close()


@torch.no_grad()
def collect_rollouts(
    task_name: str,
    source_ckpt: str,
    dataset_stats_path: str,
    output_dir: str,
    num_seed_groups: int,
    rollouts_per_seed: int,
    device: str = 'auto',
    temporal_agg: bool = True,
    deterministic: bool = False,
    decay_k: float = 0.01,
    max_timesteps: Optional[int] = None,
    source_label: str = 'bc',
    seed_start: int = 1000,
    policy_overrides: Optional[Dict[str, Any]] = None,
    sampling_strategy: str = 'auto',
    exploration_std: float = 0.05,
    deterministic_candidates: int = 0,
):
    device_obj = resolve_device(device)
    task_config = get_task_config(task_name)
    camera_names = task_config['camera_names']
    episode_len = int(max_timesteps or task_config['episode_len'])
    stats = RolloutStats.from_pickle(dataset_stats_path)
    policy = build_act_policy(task_name, device=device, overrides=policy_overrides)
    checkpoint_load_status = load_policy_checkpoint(policy, source_ckpt, device=device, strict=True)
    actual_sampling_strategy = _resolve_rollout_sampling_strategy(
        sampling_strategy,
        deterministic=deterministic,
        checkpoint_load_status=checkpoint_load_status,
    )
    policy.eval()

    output_dir = str(Path(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    save_json(
        os.path.join(output_dir, 'collection_meta.json'),
        {
            'schema_version': ROLLOUT_SCHEMA_VERSION,
            'task_name': task_name,
            'source_checkpoint': source_ckpt,
            'dataset_stats_path': dataset_stats_path,
            'camera_names': camera_names,
            'num_seed_groups': num_seed_groups,
            'rollouts_per_seed': rollouts_per_seed,
            'temporal_agg': temporal_agg,
            'deterministic': deterministic,
            'device': str(device_obj),
            'source_label': source_label,
            'sampling_strategy': actual_sampling_strategy,
            'requested_sampling_strategy': sampling_strategy,
            'exploration_std': float(exploration_std),
            'deterministic_candidates': int(deterministic_candidates),
            'checkpoint_missing_keys': list(getattr(checkpoint_load_status, 'missing_keys', [])),
            'created_at_unix': time.time(),
        },
    )

    env, init_state_pool, env_max_reward = _make_task_env(task_name, task_config)
    num_queries = policy.model.num_queries

    for seed_offset in range(num_seed_groups):
        seed = seed_start + seed_offset
        initial_pose, init_state_meta = _select_task_initial_state(task_name, init_state_pool, seed)
        for candidate_index in range(rollouts_per_seed):
            rollout_id = f'seed{seed:04d}-cand{candidate_index:02d}'
            candidate_force_deterministic = candidate_index < deterministic_candidates
            state_obj = _reset_task_env(env, task_name, initial_pose)
            history = ChunkDistributionHistory(num_queries=num_queries, temporal_agg=temporal_agg, decay_k=decay_k, detach_history=True)
            actions_norm = []
            actions_env = []
            rewards = []
            qpos_seq = []
            for t in range(episode_len):
                obs = _extract_obs(task_name, state_obj)
                qpos_raw = _extract_qpos(task_name, obs)
                qpos_seq.append(qpos_raw)
                qpos = torch.from_numpy(stats.normalize_qpos(qpos_raw)).float().to(device_obj).unsqueeze(0)
                image = get_image_from_obs(task_name, obs, camera_names, device_obj)
                dist, value = history.step_distribution(policy, qpos, image, t)
                raw_action = _sample_rollout_action(
                    dist,
                    actual_sampling_strategy,
                    exploration_std,
                    force_deterministic=candidate_force_deterministic,
                )
                raw_action_np = raw_action.squeeze(0).detach().cpu().numpy().astype(np.float32)
                env_action = stats.denormalize_action(raw_action_np).astype(np.float32)
                state_obj, reward, done = _step_task_env(env, task_name, env_action, source_action_dim=task_config.get('source_action_dim'))
                actions_norm.append(raw_action_np)
                actions_env.append(env_action)
                rewards.append(reward)
                if done:
                    break
            rewards_np = np.asarray(rewards, dtype=np.float32)
            actions_norm_np = np.asarray(actions_norm, dtype=np.float32)
            actions_env_np = np.asarray(actions_env, dtype=np.float32)
            qpos_seq_np = np.asarray(qpos_seq, dtype=np.float32)
            highest_reward = float(np.max(rewards_np)) if len(rewards_np) else 0.0
            episode_return = float(np.sum(rewards_np))
            success = bool(highest_reward >= env_max_reward)
            metadata = {
                'schema_version': ROLLOUT_SCHEMA_VERSION,
                'rollout_id': rollout_id,
                'task_name': task_name,
                'source_checkpoint': source_ckpt,
                'source_label': source_label,
                'seed': seed,
                'candidate_index': candidate_index,
                'temporal_agg': temporal_agg,
                'deterministic': deterministic,
                'sampling_strategy': actual_sampling_strategy,
                'exploration_std': float(exploration_std),
                'force_deterministic': bool(candidate_force_deterministic),
                'initial_object_pose': initial_pose.tolist(),
                'initial_state_kind': 'libero_init_state' if is_libero_task(task_name) else 'sim_object_pose',
                'num_steps': int(len(actions_env_np)),
                'episode_return': episode_return,
                'highest_reward': highest_reward,
                'success': success,
                'env_max_reward': float(env_max_reward),
                **init_state_meta,
            }
            arrays = {
                'actions_norm': actions_norm_np,
                'actions_env': actions_env_np,
                'rewards': rewards_np,
                'qpos': qpos_seq_np,
                'actions': actions_env_np,
            }
            save_rollout_record(output_dir, metadata, arrays)




def _build_action_chunk(actions_norm: np.ndarray, start: int, num_queries: int, device: torch.device):
    action_dim = int(actions_norm.shape[1])
    chunk = np.zeros((num_queries, action_dim), dtype=np.float32)
    is_pad = np.ones((num_queries,), dtype=bool)
    end = min(len(actions_norm), start + num_queries)
    valid = max(0, end - start)
    if valid > 0:
        chunk[:valid] = actions_norm[start:end]
        is_pad[:valid] = False
    chunk_tensor = torch.from_numpy(chunk).float().to(device).unsqueeze(0)
    is_pad_tensor = torch.from_numpy(is_pad).to(device).unsqueeze(0)
    return chunk_tensor, is_pad_tensor


def trajectory_chunk_score(
    policy,
    task_name: str,
    camera_names: Sequence[str],
    stats: RolloutStats,
    initial_object_pose: np.ndarray,
    actions_env: np.ndarray,
    actions_norm: np.ndarray,
    device: str = 'auto',
    window_start: int = 0,
    window_length: Optional[int] = None,
    posterior_decode_mode: str = 'mean',
    kl_coef: float = 1.0,
):
    device_obj = resolve_device(device)
    replay = TrajectoryReplay(task_name, camera_names, initial_object_pose, actions_env, task_config=get_task_config(task_name))
    total_score = None
    total_log_prob = None
    total_entropy = None
    total_kl = None
    chunk_count = 0
    token_count = 0
    end = len(actions_norm) if window_length is None else min(len(actions_norm), window_start + window_length)
    for t, obs in enumerate(replay.iter_observations()):
        if t >= end:
            break
        if t < window_start:
            continue
        qpos_raw = _extract_qpos(task_name, obs)
        qpos = torch.from_numpy(stats.normalize_qpos(qpos_raw)).float().to(device_obj).unsqueeze(0)
        image = get_image_from_obs(task_name, obs, camera_names, device_obj)
        chunk_actions, chunk_is_pad = _build_action_chunk(actions_norm, t, policy.model.num_queries, device_obj)
        chunk_stats = policy.score_action_chunk(
            qpos,
            image,
            chunk_actions,
            chunk_is_pad,
            posterior_decode_mode=posterior_decode_mode,
            kl_coef=kl_coef,
        )
        score = chunk_stats['score']
        total_score = score if total_score is None else total_score + score
        total_log_prob = chunk_stats['log_prob'] if total_log_prob is None else total_log_prob + chunk_stats['log_prob']
        total_entropy = chunk_stats['entropy'] if total_entropy is None else total_entropy + chunk_stats['entropy']
        total_kl = chunk_stats['kl'] if total_kl is None else total_kl + chunk_stats['kl']
        chunk_count += 1
        token_count += int(chunk_stats['token_count'].item())
        if t + 1 >= len(actions_norm):
            break
    if total_score is None:
        raise ValueError('No chunk scores accumulated; check window_start/window_length')
    score_normalizer = max(chunk_count, 1)
    token_normalizer = max(token_count, 1)
    return {
        'score': total_score,
        'mean_score': total_score / score_normalizer,
        'log_prob': total_log_prob,
        'mean_log_prob': total_log_prob / token_normalizer,
        'entropy': total_entropy,
        'mean_entropy': total_entropy / token_normalizer,
        'kl': total_kl,
        'mean_kl': total_kl / score_normalizer,
        'token_count': token_count,
        'chunk_count': chunk_count,
    }


def trajectory_logprob(
    policy,
    task_name: str,
    camera_names: Sequence[str],
    stats: RolloutStats,
    initial_object_pose: np.ndarray,
    actions_env: np.ndarray,
    actions_norm: np.ndarray,
    device: str = 'auto',
    temporal_agg: bool = True,
    decay_k: float = 0.01,
    window_start: int = 0,
    window_length: Optional[int] = None,
):
    device_obj = resolve_device(device)
    replay = TrajectoryReplay(task_name, camera_names, initial_object_pose, actions_env, task_config=get_task_config(task_name))
    history = ChunkDistributionHistory(
        num_queries=policy.model.num_queries,
        temporal_agg=temporal_agg,
        decay_k=decay_k,
        detach_history=False,
    )
    total_log_prob = None
    total_entropy = None
    end = len(actions_norm) if window_length is None else min(len(actions_norm), window_start + window_length)
    value_samples = []
    token_count = 0
    for t, obs in enumerate(replay.iter_observations()):
        qpos_raw = _extract_qpos(task_name, obs)
        qpos = torch.from_numpy(stats.normalize_qpos(qpos_raw)).float().to(device_obj).unsqueeze(0)
        image = get_image_from_obs(task_name, obs, camera_names, device_obj)
        dist, value = history.step_distribution(policy, qpos, image, t)
        if window_start <= t < end:
            action = torch.from_numpy(actions_norm[t]).float().to(device_obj).unsqueeze(0)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            total_log_prob = log_prob if total_log_prob is None else total_log_prob + log_prob
            total_entropy = entropy if total_entropy is None else total_entropy + entropy
            value_samples.append(value)
            token_count += int(action.shape[-1])
        if t + 1 >= len(actions_norm):
            break
    if total_log_prob is None:
        raise ValueError('No log-prob terms accumulated; check window_start/window_length')
    normalizer = max(token_count, 1)
    mean_value = torch.stack(value_samples).mean(dim=0) if value_samples else torch.zeros((1,), device=device_obj)
    return {
        'log_prob': total_log_prob,
        'mean_log_prob': total_log_prob / normalizer,
        'entropy': total_entropy,
        'mean_entropy': total_entropy / normalizer,
        'value': mean_value,
        'token_count': token_count,
    }
