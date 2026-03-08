from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from device_utils import resolve_device
from posttrain.common import append_jsonl, build_act_policy, get_task_config, load_policy_checkpoint, load_training_state, save_json, save_policy_checkpoint, save_training_state
from posttrain.rollouts import ChunkDistributionHistory, RolloutStats, get_image_from_ts, sample_initial_object_pose, set_task_initial_pose
from sim_env import make_sim_env


def _obs_to_image_tensor(obs_images: Dict[str, np.ndarray], camera_names, device):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs_images[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    return torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)


def compute_gae(rewards, values, dones, gamma: float, gae_lambda: float):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + values
    return returns.astype(np.float32), advantages.astype(np.float32)


@torch.no_grad()
def collect_policy_rollouts(policy, task_name: str, dataset_stats_path: str, num_rollouts: int, device: str = 'auto', temporal_agg: bool = True, decay_k: float = 0.01, seed_start: int = 2000, max_timesteps: int | None = None):
    device_obj = resolve_device(device)
    task_config = get_task_config(task_name)
    camera_names = task_config['camera_names']
    max_timesteps = int(max_timesteps or task_config['episode_len'])
    stats = RolloutStats.from_pickle(dataset_stats_path)
    policy.eval()

    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward
    batch = []
    for rollout_idx in range(num_rollouts):
        seed = seed_start + rollout_idx
        initial_pose = sample_initial_object_pose(task_name, seed)
        set_task_initial_pose(task_name, initial_pose)
        ts = env.reset()
        history = ChunkDistributionHistory(policy.model.num_queries, temporal_agg=temporal_agg, decay_k=decay_k, detach_history=True)
        images = []
        qpos = []
        actions_norm = []
        actions_env = []
        rewards = []
        dones = []
        old_log_probs = []
        old_values = []
        for t in range(max_timesteps):
            obs = ts.observation
            obs_images = {cam: obs['images'][cam].copy() for cam in camera_names}
            qpos_raw = np.asarray(obs['qpos'], dtype=np.float32)
            qpos_tensor = torch.from_numpy(stats.normalize_qpos(qpos_raw)).float().to(device_obj).unsqueeze(0)
            image_tensor = _obs_to_image_tensor(obs_images, camera_names, device_obj)
            dist, value = history.step_distribution(policy, qpos_tensor, image_tensor, t)
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            env_action = stats.denormalize_action(raw_action.squeeze(0).cpu().numpy().astype(np.float32))
            ts = env.step(env_action)
            reward = float(ts.reward if ts.reward is not None else 0.0)
            images.append(obs_images)
            qpos.append(qpos_raw)
            actions_norm.append(raw_action.squeeze(0).cpu().numpy().astype(np.float32))
            actions_env.append(env_action.astype(np.float32))
            rewards.append(reward)
            dones.append(0.0)
            old_log_probs.append(float(log_prob.cpu().item()))
            old_values.append(float(value.squeeze(0).cpu().item()))
        if dones:
            dones[-1] = 1.0
        rewards_np = np.asarray(rewards, dtype=np.float32)
        values_np = np.asarray(old_values, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.float32)
        returns_np, advantages_np = compute_gae(rewards_np, values_np, dones_np, gamma=0.99, gae_lambda=0.95)
        highest_reward = float(np.max(rewards_np)) if len(rewards_np) else 0.0
        batch.append({
            'seed': seed,
            'initial_object_pose': initial_pose.astype(np.float32),
            'images': images,
            'qpos': np.asarray(qpos, dtype=np.float32),
            'actions_norm': np.asarray(actions_norm, dtype=np.float32),
            'actions_env': np.asarray(actions_env, dtype=np.float32),
            'rewards': rewards_np,
            'dones': dones_np,
            'old_log_probs': np.asarray(old_log_probs, dtype=np.float32),
            'old_values': values_np,
            'returns': returns_np,
            'advantages': advantages_np,
            'episode_return': float(np.sum(rewards_np)),
            'highest_reward': highest_reward,
            'success': bool(highest_reward == env_max_reward),
            'env_max_reward': float(env_max_reward),
        })
    return batch


def evaluate_rollout_batch(policy, batch, stats: RolloutStats, task_name: str, camera_names, device: str = 'auto', temporal_agg: bool = True, decay_k: float = 0.01):
    device_obj = resolve_device(device)
    rollout_metrics = []
    for rollout in batch:
        history = ChunkDistributionHistory(policy.model.num_queries, temporal_agg=temporal_agg, decay_k=decay_k, detach_history=False)
        new_log_probs = []
        entropies = []
        values = []
        for t, (obs_images, qpos_raw, action_norm) in enumerate(zip(rollout['images'], rollout['qpos'], rollout['actions_norm'])):
            qpos_tensor = torch.from_numpy(stats.normalize_qpos(qpos_raw)).float().to(device_obj).unsqueeze(0)
            image_tensor = _obs_to_image_tensor(obs_images, camera_names, device_obj)
            dist, value = history.step_distribution(policy, qpos_tensor, image_tensor, t)
            action_tensor = torch.from_numpy(action_norm).float().to(device_obj).unsqueeze(0)
            new_log_probs.append(dist.log_prob(action_tensor).sum(dim=-1))
            entropies.append(dist.entropy().sum(dim=-1))
            values.append(value)
        rollout_metrics.append({
            'new_log_probs': torch.cat(new_log_probs, dim=0),
            'entropies': torch.cat(entropies, dim=0),
            'values': torch.cat(values, dim=0),
            'old_log_probs': torch.from_numpy(rollout['old_log_probs']).float().to(device_obj),
            'returns': torch.from_numpy(rollout['returns']).float().to(device_obj),
            'advantages': torch.from_numpy(rollout['advantages']).float().to(device_obj),
        })
    return rollout_metrics


def train_ppo_like(args):
    task_config = get_task_config(args.task_name)
    camera_names = task_config['camera_names']
    stats = RolloutStats.from_pickle(args.dataset_stats)
    policy_overrides = {'lr': args.lr} if hasattr(args, 'lr') else None
    policy = build_act_policy(args.task_name, device=args.device, overrides=policy_overrides)
    load_policy_checkpoint(policy, args.init_ckpt, device=args.device, strict=True)
    optimizer = policy.configure_optimizers()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_stats_link = out_dir / 'dataset_stats.pkl'
    if not dataset_stats_link.exists():
        dataset_stats_link.symlink_to(Path(args.dataset_stats).resolve())
    metrics_path = out_dir / 'metrics.jsonl'
    state_path = out_dir / 'ppo_state.pt'
    save_json(str(out_dir / 'config.json'), vars(args))

    start_update = 0
    best_success = -1.0
    best_return = -1e9
    if args.resume_ckpt == 'auto' and state_path.exists():
        metadata = load_training_state(str(state_path), policy, optimizer=optimizer, device=args.device)
        start_update = int(metadata.get('update', 0))
        best_success = float(metadata.get('best_success', -1.0))
        best_return = float(metadata.get('best_return', -1e9))
    elif args.resume_ckpt and args.resume_ckpt not in ('', 'auto'):
        metadata = load_training_state(args.resume_ckpt, policy, optimizer=optimizer, device=args.device)
        start_update = int(metadata.get('update', 0))
        best_success = float(metadata.get('best_success', -1.0))
        best_return = float(metadata.get('best_return', -1e9))

    pbar = tqdm(range(start_update, args.updates), desc='ppo')
    for update in pbar:
        batch = collect_policy_rollouts(
            policy,
            args.task_name,
            args.dataset_stats,
            num_rollouts=args.num_rollouts,
            device=args.device,
            temporal_agg=args.temporal_agg,
            decay_k=args.decay_k,
            seed_start=args.seed_start + update * args.num_rollouts,
            max_timesteps=args.max_timesteps,
        )
        mean_success = float(np.mean([float(r['success']) for r in batch]))
        mean_return = float(np.mean([r['episode_return'] for r in batch]))

        policy.train()
        update_metrics = []
        for _epoch in range(args.update_epochs):
            rollout_metrics = evaluate_rollout_batch(
                policy,
                batch,
                stats,
                args.task_name,
                camera_names,
                device=args.device,
                temporal_agg=args.temporal_agg,
                decay_k=args.decay_k,
            )
            policy_losses = []
            value_losses = []
            entropy_terms = []
            optimizer.zero_grad()
            for metrics in rollout_metrics:
                advantages = metrics['advantages']
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                ratios = torch.exp(metrics['new_log_probs'] - metrics['old_log_probs'])
                clipped = torch.clamp(ratios, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                policy_loss = -torch.min(ratios * advantages, clipped * advantages).mean()
                value_loss = F.mse_loss(metrics['values'], metrics['returns'])
                entropy = metrics['entropies'].mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                loss.backward()
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropy_terms.append(entropy.detach())
            clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            update_metrics.append({
                'policy_loss': float(torch.stack(policy_losses).mean().cpu().item()),
                'value_loss': float(torch.stack(value_losses).mean().cpu().item()),
                'entropy': float(torch.stack(entropy_terms).mean().cpu().item()),
            })
        mean_policy_loss = float(np.mean([m['policy_loss'] for m in update_metrics]))
        mean_value_loss = float(np.mean([m['value_loss'] for m in update_metrics]))
        mean_entropy = float(np.mean([m['entropy'] for m in update_metrics]))
        record = {
            'update': update,
            'mean_success': mean_success,
            'mean_return': mean_return,
            'policy_loss': mean_policy_loss,
            'value_loss': mean_value_loss,
            'entropy': mean_entropy,
        }
        append_jsonl(str(metrics_path), record)
        pbar.set_postfix(success=mean_success, ret=mean_return)

        if (mean_success > best_success) or (math.isclose(mean_success, best_success) and mean_return > best_return):
            best_success = mean_success
            best_return = mean_return
            save_policy_checkpoint(policy, str(out_dir / 'policy_ppo_best.ckpt'))

        save_policy_checkpoint(policy, str(out_dir / 'policy_ppo_last.ckpt'))
        if ((update + 1) % args.checkpoint_interval == 0) or (update + 1 == args.updates):
            save_policy_checkpoint(policy, str(out_dir / f'policy_ppo_update_{update + 1}.ckpt'))
        save_training_state(
            str(state_path),
            policy,
            optimizer,
            {
                'update': update + 1,
                'best_success': best_success,
                'best_return': best_return,
                'init_ckpt': args.init_ckpt,
            },
        )


def parse_args():
    parser = argparse.ArgumentParser(description='PPO-like RL fine-tuning for ACT post-training')
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--dataset_stats', required=True)
    parser.add_argument('--init_ckpt', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--updates', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_rollouts', type=int, default=2)
    parser.add_argument('--update_epochs', type=int, default=1)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    parser.add_argument('--resume_ckpt', default='auto')
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--decay_k', type=float, default=0.01)
    parser.add_argument('--seed_start', type=int, default=3000)
    parser.add_argument('--max_timesteps', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    train_ppo_like(parse_args())
