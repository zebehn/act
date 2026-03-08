#!/usr/bin/env python
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / 'detr') not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / 'detr'))

import argparse
from pathlib import Path

from posttrain.preferences import build_preference_pairs


def parse_args():
    parser = argparse.ArgumentParser(description='Build deterministic preference pairs from rollout artifacts')
    parser.add_argument('--rollout_dir', required=True)
    parser.add_argument('--out_file', required=True)
    parser.add_argument('--window_length', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    header, pairs = build_preference_pairs(args.rollout_dir, args.out_file, window_length=args.window_length)
    print(f'Built {len(pairs)} preference pairs -> {Path(args.out_file).resolve()}')
    print(header)