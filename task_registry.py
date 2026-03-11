import os
from copy import deepcopy
from typing import Any, Dict, List

from constants import SIM_TASK_CONFIGS
from libero_adapter import resolve_libero_task_config


def is_libero_task(task_name: str) -> bool:
    return task_name.startswith('libero_')


def is_sim_task(task_name: str) -> bool:
    return task_name.startswith('sim_')


def resolve_task_config(task_name: str) -> Dict[str, Any]:
    if task_name in SIM_TASK_CONFIGS:
        cfg = deepcopy(SIM_TASK_CONFIGS[task_name])
        cfg['dataset_type'] = cfg.get('dataset_type', 'episodic_hdf5')
        cfg['task_source'] = 'sim'
        cfg['state_dim'] = cfg.get('state_dim', 14)
        cfg['action_dim'] = cfg.get('action_dim', 14)
        return cfg
    if is_libero_task(task_name):
        cfg = resolve_libero_task_config(task_name)
        cfg['task_source'] = 'libero'
        return cfg
    try:
        from aloha_scripts.constants import TASK_CONFIGS
    except Exception as exc:
        raise KeyError(f'Unknown task {task_name} and aloha_scripts unavailable') from exc
    if task_name not in TASK_CONFIGS:
        raise KeyError(f'Unknown task {task_name}')
    cfg = deepcopy(TASK_CONFIGS[task_name])
    cfg['dataset_type'] = cfg.get('dataset_type', 'episodic_hdf5')
    cfg['task_source'] = 'real'
    cfg['state_dim'] = cfg.get('state_dim', 14)
    cfg['action_dim'] = cfg.get('action_dim', 14)
    return cfg


def list_task_episode_ids(task_config: Dict[str, Any]) -> List[int]:
    dataset_type = task_config.get('dataset_type', 'episodic_hdf5')
    if dataset_type == 'libero_demo_hdf5':
        from libero_adapter import list_libero_episode_ids
        return list_libero_episode_ids(task_config['dataset_path'])

    import glob
    import re
    pattern = os.path.join(task_config['dataset_dir'], 'episode_*.hdf5')
    episode_ids = []
    for path in glob.glob(pattern):
        match = re.search(r'episode_(\d+)\.hdf5$', os.path.basename(path))
        if match:
            episode_ids.append(int(match.group(1)))
    return sorted(set(episode_ids))
