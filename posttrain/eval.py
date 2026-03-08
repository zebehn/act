from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from posttrain.rollouts import collect_rollouts
from posttrain.schema import iter_rollout_metadata


def summarize_rollout_dir(rollout_dir: str) -> Dict[str, float]:
    records = list(iter_rollout_metadata(rollout_dir))
    if not records:
        raise ValueError(f'No rollout records found in {rollout_dir}')
    success = np.mean([float(r['success']) for r in records])
    avg_return = np.mean([float(r['episode_return']) for r in records])
    max_reward = int(max(float(r['env_max_reward']) for r in records))
    reward_thresholds = {}
    for r in range(max_reward + 1):
        count = sum(float(record['highest_reward']) >= r for record in records)
        reward_thresholds[r] = count / len(records)
    return {
        'success_rate': float(success),
        'average_return': float(avg_return),
        'rollout_count': len(records),
        'max_reward': max_reward,
        'reward_thresholds': reward_thresholds,
    }


def evaluate_checkpoint(args):
    collect_rollouts(
        task_name=args.task_name,
        source_ckpt=args.source_ckpt,
        dataset_stats_path=args.dataset_stats,
        output_dir=args.out_dir,
        num_seed_groups=args.num_rollouts,
        rollouts_per_seed=1,
        device=args.device,
        temporal_agg=args.temporal_agg,
        deterministic=not args.sample_actions,
        decay_k=args.decay_k,
        max_timesteps=args.max_timesteps,
        source_label=args.source_label,
        seed_start=args.seed_start,
    )
    summary = summarize_rollout_dir(args.out_dir)
    summary_path = Path(args.out_dir) / 'result_summary.json'
    summary_path.write_text(__import__('json').dumps(summary, indent=2, sort_keys=True))
    print(f"success_rate={summary['success_rate']} average_return={summary['average_return']} rollouts={summary['rollout_count']}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint via simulator rollouts for post-training workflows')
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--source_ckpt', required=True)
    parser.add_argument('--dataset_stats', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--num_rollouts', type=int, default=10)
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--sample_actions', action='store_true')
    parser.add_argument('--decay_k', type=float, default=0.01)
    parser.add_argument('--max_timesteps', type=int, default=None)
    parser.add_argument('--source_label', default='eval')
    parser.add_argument('--seed_start', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    evaluate_checkpoint(parse_args())
