import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from task_registry import resolve_task_config
from device_utils import resolve_device
from policy import ACTPolicy

DEFAULT_ACT_HPARAMS = {
    'lr': 1e-5,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
}


def get_task_config(task_name: str) -> Dict[str, Any]:
    return resolve_task_config(task_name)


def make_act_policy_config(task_name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    task_config = get_task_config(task_name)
    cfg = dict(DEFAULT_ACT_HPARAMS)
    cfg['camera_names'] = task_config['camera_names']
    if overrides:
        cfg.update(overrides)
    return cfg


def build_act_policy(task_name: str, device: str = 'auto', overrides: Optional[Dict[str, Any]] = None) -> ACTPolicy:
    cfg = make_act_policy_config(task_name, overrides=overrides)
    cfg['device'] = device
    policy = ACTPolicy(cfg)
    policy.to(resolve_device(device))
    return policy


def unwrap_state_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        return obj['model_state_dict']
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    return obj


def load_policy_checkpoint(policy: ACTPolicy, ckpt_path: str, device: str = 'auto', strict: bool = True):
    resolved_device = resolve_device(device)
    state_obj = torch.load(ckpt_path, map_location=resolved_device)
    state_dict = unwrap_state_dict(state_obj)
    return policy.load_state_dict(state_dict, strict=strict)


def save_policy_checkpoint(policy: ACTPolicy, ckpt_path: str):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(policy.state_dict(), ckpt_path)


def save_training_state(state_path: str, policy: ACTPolicy, optimizer: torch.optim.Optimizer, metadata: Dict[str, Any]):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    payload = {
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata,
    }
    torch.save(payload, state_path)


def load_training_state(state_path: str, policy: ACTPolicy, optimizer: Optional[torch.optim.Optimizer] = None, device: str = 'auto') -> Dict[str, Any]:
    resolved_device = resolve_device(device)
    payload = torch.load(state_path, map_location=resolved_device)
    policy.load_state_dict(payload['model_state_dict'], strict=True)
    if optimizer is not None and 'optimizer_state_dict' in payload:
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    return payload.get('metadata', {})


def append_jsonl(path: str, record: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def read_jsonl(path: str):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_json(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def default_posttrain_root(task_name: str) -> str:
    return str(Path('posttrain_artifacts') / task_name)
