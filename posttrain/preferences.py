from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

from posttrain.common import save_json
from posttrain.schema import PREFERENCE_SCHEMA_VERSION, iter_rollout_metadata, pair_id, save_preference_pairs, stable_score_key


def _stable_window_start(pair_identifier: str, max_start: int) -> int:
    if max_start <= 0:
        return 0
    digest = hashlib.sha256(pair_identifier.encode('utf-8')).hexdigest()
    return int(digest[:8], 16) % (max_start + 1)


def build_preference_pairs(rollout_dir: str, output_path: str, window_length: int = 32, source_label: str = 'dpo_pairs') -> Tuple[Dict, List[Dict]]:
    records = list(iter_rollout_metadata(rollout_dir))
    collection_meta_path = os.path.join(rollout_dir, 'collection_meta.json')
    collection_meta = {}
    if os.path.exists(collection_meta_path):
        with open(collection_meta_path, 'r') as f:
            collection_meta = __import__('json').load(f)
    expected_task = collection_meta.get('task_name')
    expected_source_checkpoint = collection_meta.get('source_checkpoint')
    expected_source_label = collection_meta.get('source_label')
    groups: Dict[int, List[Dict]] = {}
    for record in records:
        if expected_task and record.get('task_name') != expected_task:
            raise ValueError(f'Rollout task mismatch inside rollout_dir: expected {expected_task}, got {record.get("task_name")}')
        if expected_source_checkpoint and record.get('source_checkpoint') != expected_source_checkpoint:
            raise ValueError('Rollout source_checkpoint mismatch inside rollout_dir')
        if expected_source_label and record.get('source_label') != expected_source_label:
            raise ValueError('Rollout source_label mismatch inside rollout_dir')
        groups.setdefault(int(record['seed']), []).append(record)

    pairs: List[Dict] = []
    skipped_groups = []
    for seed, group in sorted(groups.items()):
        ordered = sorted(group, key=stable_score_key)
        chosen = ordered[-1]
        rejected = ordered[0]
        chosen_rank = stable_score_key(chosen)
        rejected_rank = stable_score_key(rejected)
        if chosen_rank == rejected_rank:
            skipped_groups.append({'seed': seed, 'reason': 'identical_score'})
            continue
        common_steps = min(int(chosen['num_steps']), int(rejected['num_steps']))
        effective_window = min(window_length, common_steps)
        pair_identifier = pair_id(chosen['rollout_id'], rejected['rollout_id'])
        window_start = _stable_window_start(pair_identifier, max(0, common_steps - effective_window))
        pairs.append({
            'schema_version': PREFERENCE_SCHEMA_VERSION,
            'pair_id': pair_identifier,
            'task_name': chosen['task_name'],
            'seed': seed,
            'chosen_rollout_id': chosen['rollout_id'],
            'rejected_rollout_id': rejected['rollout_id'],
            'chosen_success': bool(chosen['success']),
            'rejected_success': bool(rejected['success']),
            'chosen_return': float(chosen['episode_return']),
            'rejected_return': float(rejected['episode_return']),
            'window_start': int(window_start),
            'window_length': int(effective_window),
            'ranking_rule': 'success_first_return_tiebreak',
        })

    header = {
        'schema_version': PREFERENCE_SCHEMA_VERSION,
        'rollout_dir': str(Path(rollout_dir).resolve()),
        'task_name': collection_meta.get('task_name'),
        'camera_names': collection_meta.get('camera_names', ['top']),
        'dataset_stats_path': collection_meta.get('dataset_stats_path'),
        'source_checkpoint': collection_meta.get('source_checkpoint'),
        'rollout_source_label': collection_meta.get('source_label'),
        'pair_count': len(pairs),
        'window_length': int(window_length),
        'source_label': source_label,
        'group_count': len(groups),
        'skipped_groups': skipped_groups,
        'ranking_rule': 'success_first_return_tiebreak',
    }
    save_preference_pairs(output_path, header, pairs)
    return header, pairs
