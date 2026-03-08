from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROLLOUT_SCHEMA_VERSION = 1
PREFERENCE_SCHEMA_VERSION = 1

REQUIRED_ROLLOUT_META_KEYS = {
    'schema_version',
    'rollout_id',
    'task_name',
    'source_checkpoint',
    'source_label',
    'seed',
    'candidate_index',
    'temporal_agg',
    'deterministic',
    'initial_object_pose',
    'num_steps',
    'episode_return',
    'highest_reward',
    'success',
    'env_max_reward',
}

REQUIRED_ROLLOUT_ARRAY_KEYS = {
    'actions',
    'actions_norm',
    'actions_env',
    'rewards',
    'qpos',
}


def _append_jsonl(path: str, record: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def _save_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def _read_jsonl(path: str):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rollout_paths(root_dir: str, rollout_id: str) -> Tuple[str, str]:
    base = Path(root_dir)
    return str(base / f'{rollout_id}.json'), str(base / f'{rollout_id}.npz')


def save_rollout_record(root_dir: str, metadata: Dict, arrays: Dict[str, np.ndarray]):
    os.makedirs(root_dir, exist_ok=True)
    missing_meta = REQUIRED_ROLLOUT_META_KEYS - set(metadata.keys())
    missing_arrays = REQUIRED_ROLLOUT_ARRAY_KEYS - set(arrays.keys())
    if missing_meta:
        raise KeyError(f'Missing rollout metadata keys: {sorted(missing_meta)}')
    if missing_arrays:
        raise KeyError(f'Missing rollout array keys: {sorted(missing_arrays)}')
    meta_path, npz_path = rollout_paths(root_dir, metadata['rollout_id'])
    num_steps = int(metadata['num_steps'])
    for key in ['actions', 'actions_norm', 'actions_env', 'rewards', 'qpos']:
        arr = arrays[key]
        if len(arr) != num_steps:
            raise ValueError(f'Rollout array length mismatch for {key}: expected {num_steps}, got {len(arr)}')
    if arrays['actions_norm'].shape != arrays['actions_env'].shape:
        raise ValueError('actions_norm and actions_env shapes must match')
    if arrays['actions'].shape != arrays['actions_env'].shape:
        raise ValueError('actions and actions_env shapes must match')
    _save_json(meta_path, metadata)
    np.savez_compressed(npz_path, **arrays)
    _append_jsonl(os.path.join(root_dir, 'manifest.jsonl'), metadata)


def load_rollout_metadata(root_dir: str, rollout_id: str) -> Dict:
    meta_path, _ = rollout_paths(root_dir, rollout_id)
    return _load_json(meta_path)


def load_rollout_arrays(root_dir: str, rollout_id: str) -> Dict[str, np.ndarray]:
    _, npz_path = rollout_paths(root_dir, rollout_id)
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def iter_rollout_metadata(root_dir: str) -> Iterable[Dict]:
    manifest = os.path.join(root_dir, 'manifest.jsonl')
    if os.path.exists(manifest):
        yield from _read_jsonl(manifest)
        return
    for json_path in sorted(Path(root_dir).glob('*.json')):
        yield _load_json(str(json_path))


def stable_score_key(metadata: Dict):
    return (
        int(bool(metadata['success'])),
        float(metadata['episode_return']),
        float(metadata['highest_reward']),
        metadata['rollout_id'],
    )


def pair_id(chosen_rollout_id: str, rejected_rollout_id: str) -> str:
    return f'{chosen_rollout_id}__vs__{rejected_rollout_id}'


def save_preference_pairs(output_path: str, header: Dict, pairs: List[Dict]):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    _save_json(output_path + '.meta.json', header)


def load_preference_pairs(path: str) -> Tuple[Dict, List[Dict]]:
    header = _load_json(path + '.meta.json')
    with open(path, 'r') as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    return header, pairs
