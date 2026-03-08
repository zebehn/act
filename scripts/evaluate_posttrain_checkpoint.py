#!/usr/bin/env python
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / 'detr') not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / 'detr'))

from posttrain.eval import evaluate_checkpoint, parse_args

if __name__ == '__main__':
    evaluate_checkpoint(parse_args())