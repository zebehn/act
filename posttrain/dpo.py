from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from posttrain.common import build_act_policy, load_policy_checkpoint, load_training_state, save_policy_checkpoint, save_training_state, append_jsonl, save_json, read_jsonl
from posttrain.rollouts import RolloutStats, trajectory_logprob
from posttrain.schema import load_preference_pairs, load_rollout_arrays, load_rollout_metadata


def dpo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta: float):
    margin = beta * ((chosen_logp - rejected_logp) - (ref_chosen_logp - ref_rejected_logp))
    loss = torch.logaddexp(torch.zeros_like(margin), -margin).mean()
    return loss, margin.mean().detach()


def _resolve_pair_rollouts(header: Dict[str, Any], pair: Dict[str, Any]):
    rollout_dir = header['rollout_dir']
    chosen_meta = load_rollout_metadata(rollout_dir, pair['chosen_rollout_id'])
    chosen_arrays = load_rollout_arrays(rollout_dir, pair['chosen_rollout_id'])
    rejected_meta = load_rollout_metadata(rollout_dir, pair['rejected_rollout_id'])
    rejected_arrays = load_rollout_arrays(rollout_dir, pair['rejected_rollout_id'])
    return rollout_dir, chosen_meta, chosen_arrays, rejected_meta, rejected_arrays


def train_dpo(args):
    header, pairs = load_preference_pairs(args.pref_file)
    if not pairs:
        raise ValueError('No preference pairs found for DPO training')
    header_task = header.get('task_name')
    if header_task and header_task != args.task_name:
        raise ValueError(f'Preference file task mismatch: header={header_task} args={args.task_name}')
    header_stats = header.get('dataset_stats_path')
    if header_stats and str(Path(header_stats).resolve()) != str(Path(args.dataset_stats).resolve()):
        raise ValueError(f'Dataset stats mismatch: header={header_stats} args={args.dataset_stats}')
    header_source_ckpt = header.get('source_checkpoint')
    expected_source_ckpt = str(Path(args.reference_ckpt or args.init_ckpt).resolve())
    if header_source_ckpt and str(Path(header_source_ckpt).resolve()) != expected_source_ckpt:
        raise ValueError(f'Preference source checkpoint mismatch: header={header_source_ckpt} expected={expected_source_ckpt}')

    stats = RolloutStats.from_pickle(args.dataset_stats)
    policy_overrides = {'lr': args.lr} if hasattr(args, 'lr') else None
    train_policy = build_act_policy(args.task_name, device=args.device, overrides=policy_overrides)
    ref_policy = build_act_policy(args.task_name, device=args.device, overrides=policy_overrides)
    load_policy_checkpoint(train_policy, args.init_ckpt, device=args.device, strict=True)
    load_policy_checkpoint(ref_policy, args.reference_ckpt or args.init_ckpt, device=args.device, strict=True)
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad_(False)

    optimizer = train_policy.configure_optimizers()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_stats_link = out_dir / 'dataset_stats.pkl'
    if not dataset_stats_link.exists():
        dataset_stats_link.symlink_to(Path(args.dataset_stats).resolve())
    metrics_path = out_dir / 'metrics.jsonl'
    state_path = out_dir / 'dpo_state.pt'
    config_path = out_dir / 'config.json'
    save_json(str(config_path), vars(args))

    start_step = 0
    best_loss = math.inf
    if args.resume_ckpt == 'auto' and state_path.exists():
        metadata = load_training_state(str(state_path), train_policy, optimizer=optimizer, device=args.device)
        start_step = int(metadata.get('step', 0))
        best_loss = float(metadata.get('best_loss', math.inf))
    elif args.resume_ckpt and args.resume_ckpt not in ('', 'auto'):
        metadata = load_training_state(args.resume_ckpt, train_policy, optimizer=optimizer, device=args.device)
        start_step = int(metadata.get('step', 0))
        best_loss = float(metadata.get('best_loss', math.inf))

    train_policy.train()
    pbar = tqdm(range(start_step, args.max_steps), desc='dpo')
    for step in pbar:
        pair = pairs[step % len(pairs)]
        rollout_dir, chosen_meta, chosen_arrays, rejected_meta, rejected_arrays = _resolve_pair_rollouts(header, pair)
        window_start = int(pair['window_start'])
        window_length = int(pair['window_length'])

        chosen_stats = trajectory_logprob(
            train_policy,
            args.task_name,
            header.get('camera_names', ['top']),
            stats,
            chosen_meta['initial_object_pose'],
            chosen_arrays['actions_env'],
            chosen_arrays['actions_norm'],
            device=args.device,
            temporal_agg=args.temporal_agg,
            decay_k=args.decay_k,
            window_start=window_start,
            window_length=window_length,
        )
        rejected_stats = trajectory_logprob(
            train_policy,
            args.task_name,
            header.get('camera_names', ['top']),
            stats,
            rejected_meta['initial_object_pose'],
            rejected_arrays['actions_env'],
            rejected_arrays['actions_norm'],
            device=args.device,
            temporal_agg=args.temporal_agg,
            decay_k=args.decay_k,
            window_start=window_start,
            window_length=window_length,
        )
        with torch.no_grad():
            ref_chosen_stats = trajectory_logprob(
                ref_policy,
                args.task_name,
                header.get('camera_names', ['top']),
                stats,
                chosen_meta['initial_object_pose'],
                chosen_arrays['actions_env'],
                chosen_arrays['actions_norm'],
                device=args.device,
                temporal_agg=args.temporal_agg,
                decay_k=args.decay_k,
                window_start=window_start,
                window_length=window_length,
            )
            ref_rejected_stats = trajectory_logprob(
                ref_policy,
                args.task_name,
                header.get('camera_names', ['top']),
                stats,
                rejected_meta['initial_object_pose'],
                rejected_arrays['actions_env'],
                rejected_arrays['actions_norm'],
                device=args.device,
                temporal_agg=args.temporal_agg,
                decay_k=args.decay_k,
                window_start=window_start,
                window_length=window_length,
            )

        loss, margin = dpo_loss(
            chosen_stats['log_prob'],
            rejected_stats['log_prob'],
            ref_chosen_stats['log_prob'],
            ref_rejected_stats['log_prob'],
            args.beta,
        )

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(train_policy.parameters(), args.grad_clip)
        optimizer.step()

        metrics = {
            'step': step,
            'loss': float(loss.detach().cpu().item()),
            'margin': float(margin.cpu().item()),
            'chosen_log_prob': float(chosen_stats['log_prob'].detach().cpu().item()),
            'rejected_log_prob': float(rejected_stats['log_prob'].detach().cpu().item()),
        }
        append_jsonl(str(metrics_path), metrics)
        pbar.set_postfix(loss=metrics['loss'], margin=metrics['margin'])

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_policy_checkpoint(train_policy, str(out_dir / 'policy_pref_best.ckpt'))

        if ((step + 1) % args.checkpoint_interval == 0) or (step + 1 == args.max_steps):
            save_policy_checkpoint(train_policy, str(out_dir / 'policy_pref_last.ckpt'))
            save_policy_checkpoint(train_policy, str(out_dir / f'policy_pref_step_{step + 1}.ckpt'))
            save_training_state(
                str(state_path),
                train_policy,
                optimizer,
                {
                    'step': step + 1,
                    'best_loss': best_loss,
                    'init_ckpt': args.init_ckpt,
                    'reference_ckpt': args.reference_ckpt or args.init_ckpt,
                },
            )

    if not (out_dir / 'policy_pref_last.ckpt').exists():
        save_policy_checkpoint(train_policy, str(out_dir / 'policy_pref_last.ckpt'))


def parse_args():
    parser = argparse.ArgumentParser(description='DPO-style preference optimization for ACT post-training')
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--pref_file', required=True)
    parser.add_argument('--dataset_stats', required=True)
    parser.add_argument('--init_ckpt', required=True)
    parser.add_argument('--reference_ckpt', default=None)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--max_steps', type=int, default=10)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    parser.add_argument('--resume_ckpt', default='auto')
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--decay_k', type=float, default=0.01)
    return parser.parse_args()


if __name__ == '__main__':
    train_dpo(parse_args())
