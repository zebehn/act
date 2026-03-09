#!/usr/bin/env python
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / 'detr') not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / 'detr'))

import argparse
import os
from pathlib import Path

from posttrain.rollouts import collect_rollouts


def parse_args():
    parser = argparse.ArgumentParser(description='Collect simulator rollouts for ACT post-training')
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--source_ckpt', required=True)
    parser.add_argument('--dataset_stats', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--num_seed_groups', type=int, default=24)
    parser.add_argument('--rollouts_per_seed', type=int, default=4)
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--sampling_strategy', default='auto', choices=['auto', 'model', 'mean_gaussian', 'deterministic'])
    parser.add_argument('--exploration_std', type=float, default=0.05)
    parser.add_argument('--deterministic_candidates', type=int, default=0)
    parser.add_argument('--decay_k', type=float, default=0.01)
    parser.add_argument('--max_timesteps', type=int, default=None)
    parser.add_argument('--source_label', default='bc')
    parser.add_argument('--seed_start', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    collect_rollouts(
        task_name=args.task_name,
        source_ckpt=args.source_ckpt,
        dataset_stats_path=args.dataset_stats,
        output_dir=args.out_dir,
        num_seed_groups=args.num_seed_groups,
        rollouts_per_seed=args.rollouts_per_seed,
        device=args.device,
        temporal_agg=args.temporal_agg,
        deterministic=args.deterministic,
        sampling_strategy=args.sampling_strategy,
        exploration_std=args.exploration_std,
        deterministic_candidates=args.deterministic_candidates,
        decay_k=args.decay_k,
        max_timesteps=args.max_timesteps,
        source_label=args.source_label,
        seed_start=args.seed_start,
    )
    print(f'Wrote rollout dataset to {Path(args.out_dir).resolve()}')