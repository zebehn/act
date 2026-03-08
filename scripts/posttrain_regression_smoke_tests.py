#!/usr/bin/env python3
import os
import tempfile

import numpy as np
import torch

import posttrain.dpo as dpo_mod
import posttrain.ppo as ppo_mod
from posttrain.preferences import build_preference_pairs
from posttrain.schema import load_preference_pairs, load_rollout_arrays, load_rollout_metadata, save_rollout_record


class _DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        self.model = type('DummyModel', (), {'num_queries': 4})()

    def to(self, _device):
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def eval(self):
        super().eval()
        return self

    def state_dict(self):
        return {'w': self.w.detach().clone()}

    def load_state_dict(self, state_dict, strict=True):
        self.w.data.copy_(state_dict['w'])
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def configure_optimizers(self):
        return self.optimizer


class _DummyStats:
    def normalize_qpos(self, qpos):
        return qpos


def _write_rollout(root_dir, rollout_id, seed, success, episode_return, highest_reward, num_steps=8, source_checkpoint='/tmp/fake.ckpt', source_label='bc'):
    metadata = {
        'schema_version': 1,
        'rollout_id': rollout_id,
        'task_name': 'sim_transfer_cube_scripted',
        'source_checkpoint': source_checkpoint,
        'source_label': source_label,
        'seed': seed,
        'candidate_index': int(rollout_id.split('cand')[-1]),
        'temporal_agg': True,
        'deterministic': False,
        'initial_object_pose': [0.1, 0.5, 0.05, 1, 0, 0, 0],
        'num_steps': num_steps,
        'episode_return': float(episode_return),
        'highest_reward': float(highest_reward),
        'success': bool(success),
        'env_max_reward': 4.0,
    }
    arrays = {
        'actions': np.zeros((num_steps, 14), dtype=np.float32),
        'actions_norm': np.zeros((num_steps, 14), dtype=np.float32),
        'actions_env': np.zeros((num_steps, 14), dtype=np.float32),
        'rewards': np.linspace(0, highest_reward, num_steps, dtype=np.float32),
        'qpos': np.zeros((num_steps, 14), dtype=np.float32),
    }
    save_rollout_record(root_dir, metadata, arrays)


def test_rollout_artifact_round_trip_contains_required_metadata():
    with tempfile.TemporaryDirectory() as td:
        _write_rollout(td, 'seed1000-cand00', 1000, True, 10.0, 4.0)
        meta = load_rollout_metadata(td, 'seed1000-cand00')
        arrays = load_rollout_arrays(td, 'seed1000-cand00')
        assert meta['success'] is True
        assert meta['seed'] == 1000
        assert meta['temporal_agg'] is True
        assert meta['initial_object_pose']
        assert arrays['actions_env'].shape == (8, 14)
        assert arrays['rewards'].shape == (8,)


def test_preference_pairs_rank_success_before_return():
    with tempfile.TemporaryDirectory() as td:
        # group 1: success should dominate return
        _write_rollout(td, 'seed1000-cand00', 1000, False, 100.0, 3.0)
        _write_rollout(td, 'seed1000-cand01', 1000, True, 20.0, 4.0)
        # group 2: tie on success, return breaks tie
        _write_rollout(td, 'seed1001-cand00', 1001, True, 10.0, 4.0)
        _write_rollout(td, 'seed1001-cand01', 1001, True, 30.0, 4.0)
        pref_path = os.path.join(td, 'prefs.jsonl')
        header, pairs = build_preference_pairs(td, pref_path, window_length=4)
        assert len(pairs) == 2
        p0 = [p for p in pairs if p['seed'] == 1000][0]
        assert p0['chosen_rollout_id'] == 'seed1000-cand01'
        p1 = [p for p in pairs if p['seed'] == 1001][0]
        assert p1['chosen_return'] == 30.0
        header2, pairs2 = load_preference_pairs(pref_path)
        assert pairs == pairs2
        assert header2['pair_count'] == 2



def test_rollout_schema_rejects_missing_required_arrays():
    with tempfile.TemporaryDirectory() as td:
        metadata = {
            'schema_version': 1,
            'rollout_id': 'bad-rollout',
            'task_name': 'sim_transfer_cube_scripted',
            'source_checkpoint': '/tmp/fake.ckpt',
            'source_label': 'bc',
            'seed': 1000,
            'candidate_index': 0,
            'temporal_agg': True,
            'deterministic': False,
            'initial_object_pose': [0.1, 0.5, 0.05, 1, 0, 0, 0],
            'num_steps': 2,
            'episode_return': 0.0,
            'highest_reward': 0.0,
            'success': False,
            'env_max_reward': 4.0,
        }
        arrays = {
            'actions': np.zeros((2, 14), dtype=np.float32),
            'rewards': np.zeros((2,), dtype=np.float32),
        }
        try:
            save_rollout_record(td, metadata, arrays)
        except KeyError:
            return
        raise AssertionError('Expected save_rollout_record to reject missing arrays')


def test_dpo_rejects_preference_dataset_stats_mismatch():
    with tempfile.TemporaryDirectory() as td:
        rollout_dir = os.path.join(td, 'rollouts')
        os.makedirs(rollout_dir, exist_ok=True)
        _write_rollout(rollout_dir, 'seed1000-cand00', 1000, False, 1.0, 1.0, source_checkpoint=os.path.join(td, 'correct_init.ckpt'))
        _write_rollout(rollout_dir, 'seed1000-cand01', 1000, True, 2.0, 4.0, source_checkpoint=os.path.join(td, 'correct_init.ckpt'))
        with open(os.path.join(rollout_dir, 'collection_meta.json'), 'w') as f:
            import json
            json.dump({'task_name': 'sim_transfer_cube_scripted', 'camera_names': ['top'], 'dataset_stats_path': os.path.join(td, 'correct_stats.pkl'), 'source_checkpoint': os.path.join(td, 'correct_init.ckpt'), 'source_label': 'bc'}, f)
        pref_path = os.path.join(td, 'prefs.jsonl')
        build_preference_pairs(rollout_dir, pref_path, window_length=4)
        import pickle
        with open(os.path.join(td, 'wrong_stats.pkl'), 'wb') as f:
            pickle.dump({'qpos_mean': np.zeros(14, dtype=np.float32), 'qpos_std': np.ones(14, dtype=np.float32), 'action_mean': np.zeros(14, dtype=np.float32), 'action_std': np.ones(14, dtype=np.float32)}, f)
        torch.save({'dummy': True}, os.path.join(td, 'init.ckpt'))
        torch.save({'dummy': True}, os.path.join(td, 'correct_init.ckpt'))
        orig_build = dpo_mod.build_act_policy
        orig_load = dpo_mod.load_policy_checkpoint
        try:
            dpo_mod.build_act_policy = lambda *_args, **_kwargs: _DummyPolicy()
            dpo_mod.load_policy_checkpoint = lambda *_args, **_kwargs: None
            from types import SimpleNamespace
            args = SimpleNamespace(
                task_name='sim_transfer_cube_scripted',
                pref_file=pref_path,
                dataset_stats=os.path.join(td, 'wrong_stats.pkl'),
                init_ckpt=os.path.join(td, 'init.ckpt'),
                reference_ckpt=None,
                out_dir=os.path.join(td, 'dpo_out'),
                device='cpu',
                max_steps=1,
                beta=0.1,
                lr=1e-5,
                grad_clip=1.0,
                checkpoint_interval=1,
                resume_ckpt='auto',
                temporal_agg=True,
                decay_k=0.01,
            )
            try:
                dpo_mod.train_dpo(args)
            except ValueError:
                return
            raise AssertionError('Expected DPO trainer to reject dataset_stats mismatch')
        finally:
            dpo_mod.build_act_policy = orig_build
            dpo_mod.load_policy_checkpoint = orig_load


def test_rollout_schema_rejects_num_steps_mismatch():
    with tempfile.TemporaryDirectory() as td:
        metadata = {
            'schema_version': 1,
            'rollout_id': 'bad-steps',
            'task_name': 'sim_transfer_cube_scripted',
            'source_checkpoint': '/tmp/fake.ckpt',
            'source_label': 'bc',
            'seed': 1000,
            'candidate_index': 0,
            'temporal_agg': True,
            'deterministic': False,
            'initial_object_pose': [0.1, 0.5, 0.05, 1, 0, 0, 0],
            'num_steps': 3,
            'episode_return': 0.0,
            'highest_reward': 0.0,
            'success': False,
            'env_max_reward': 4.0,
        }
        arrays = {
            'actions': np.zeros((2, 14), dtype=np.float32),
            'actions_norm': np.zeros((2, 14), dtype=np.float32),
            'actions_env': np.zeros((2, 14), dtype=np.float32),
            'rewards': np.zeros((2,), dtype=np.float32),
            'qpos': np.zeros((2, 14), dtype=np.float32),
        }
        try:
            save_rollout_record(td, metadata, arrays)
        except ValueError:
            return
        raise AssertionError('Expected save_rollout_record to reject num_steps mismatch')


def test_rollout_schema_rejects_action_shape_mismatch():
    with tempfile.TemporaryDirectory() as td:
        metadata = {
            'schema_version': 1,
            'rollout_id': 'bad-shape',
            'task_name': 'sim_transfer_cube_scripted',
            'source_checkpoint': '/tmp/fake.ckpt',
            'source_label': 'bc',
            'seed': 1000,
            'candidate_index': 0,
            'temporal_agg': True,
            'deterministic': False,
            'initial_object_pose': [0.1, 0.5, 0.05, 1, 0, 0, 0],
            'num_steps': 2,
            'episode_return': 0.0,
            'highest_reward': 0.0,
            'success': False,
            'env_max_reward': 4.0,
        }
        arrays = {
            'actions': np.zeros((2, 14), dtype=np.float32),
            'actions_norm': np.zeros((2, 14), dtype=np.float32),
            'actions_env': np.zeros((2, 13), dtype=np.float32),
            'rewards': np.zeros((2,), dtype=np.float32),
            'qpos': np.zeros((2, 14), dtype=np.float32),
        }
        try:
            save_rollout_record(td, metadata, arrays)
        except ValueError:
            return
        raise AssertionError('Expected save_rollout_record to reject action shape mismatch')

def test_dpo_rejects_preference_source_checkpoint_mismatch():
    with tempfile.TemporaryDirectory() as td:
        rollout_dir = os.path.join(td, 'rollouts')
        os.makedirs(rollout_dir, exist_ok=True)
        _write_rollout(rollout_dir, 'seed1000-cand00', 1000, False, 1.0, 1.0, source_checkpoint=os.path.join(td, 'correct_init.ckpt'))
        _write_rollout(rollout_dir, 'seed1000-cand01', 1000, True, 2.0, 4.0, source_checkpoint=os.path.join(td, 'correct_init.ckpt'))
        with open(os.path.join(rollout_dir, 'collection_meta.json'), 'w') as f:
            import json
            json.dump({'task_name': 'sim_transfer_cube_scripted', 'camera_names': ['top'], 'dataset_stats_path': os.path.join(td, 'stats.pkl'), 'source_checkpoint': os.path.join(td, 'correct_init.ckpt'), 'source_label': 'bc'}, f)
        pref_path = os.path.join(td, 'prefs.jsonl')
        build_preference_pairs(rollout_dir, pref_path, window_length=4)
        import pickle
        with open(os.path.join(td, 'stats.pkl'), 'wb') as f:
            pickle.dump({'qpos_mean': np.zeros(14, dtype=np.float32), 'qpos_std': np.ones(14, dtype=np.float32), 'action_mean': np.zeros(14, dtype=np.float32), 'action_std': np.ones(14, dtype=np.float32)}, f)
        torch.save({'dummy': True}, os.path.join(td, 'wrong_init.ckpt'))
        torch.save({'dummy': True}, os.path.join(td, 'correct_init.ckpt'))
        orig_build = dpo_mod.build_act_policy
        orig_load = dpo_mod.load_policy_checkpoint
        try:
            dpo_mod.build_act_policy = lambda *_args, **_kwargs: _DummyPolicy()
            dpo_mod.load_policy_checkpoint = lambda *_args, **_kwargs: None
            from types import SimpleNamespace
            args = SimpleNamespace(
                task_name='sim_transfer_cube_scripted',
                pref_file=pref_path,
                dataset_stats=os.path.join(td, 'stats.pkl'),
                init_ckpt=os.path.join(td, 'wrong_init.ckpt'),
                reference_ckpt=None,
                out_dir=os.path.join(td, 'dpo_out'),
                device='cpu',
                max_steps=1,
                beta=0.1,
                lr=1e-5,
                grad_clip=1.0,
                checkpoint_interval=1,
                resume_ckpt='auto',
                temporal_agg=True,
                decay_k=0.01,
            )
            try:
                dpo_mod.train_dpo(args)
            except ValueError:
                return
            raise AssertionError('Expected DPO trainer to reject source checkpoint mismatch')
        finally:
            dpo_mod.build_act_policy = orig_build
            dpo_mod.load_policy_checkpoint = orig_load


def test_posttrain_policy_loads_legacy_bc_checkpoint_with_expected_missing_keys_only():
    import sys
    sys.argv = ['x', '--ckpt_dir', '/tmp/x', '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1', '--device', 'cpu']
    from policy import ACTPolicy
    cfg = {
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
        'camera_names': ['top'],
        'device': 'cpu',
    }
    policy = ACTPolicy(cfg)
    state = policy.state_dict()
    legacy = {k: v for k, v in state.items() if k not in {'model.action_log_std', 'model.value_head.weight', 'model.value_head.bias'}}
    load_result = policy.load_state_dict(legacy, strict=True)
    assert set(load_result.missing_keys) <= {'model.action_log_std', 'model.value_head.weight', 'model.value_head.bias'}
    bad = dict(legacy)
    bad.pop('model.action_head.bias')
    try:
        policy.load_state_dict(bad, strict=True)
    except RuntimeError:
        return
    raise AssertionError('Expected strict compatibility load to reject unexpected missing keys')


def test_dpo_one_step_smoke_writes_checkpoint_and_log():
    with tempfile.TemporaryDirectory() as td:
        rollout_dir = os.path.join(td, 'rollouts')
        os.makedirs(rollout_dir, exist_ok=True)
        _write_rollout(rollout_dir, 'seed1000-cand00', 1000, False, 1.0, 1.0)
        _write_rollout(rollout_dir, 'seed1000-cand01', 1000, True, 2.0, 4.0)
        # minimal collection meta for camera names + stats path
        with open(os.path.join(rollout_dir, 'collection_meta.json'), 'w') as f:
            import json
            json.dump({'task_name': 'sim_transfer_cube_scripted', 'camera_names': ['top'], 'dataset_stats_path': os.path.join(td, 'stats.pkl')}, f)
        pref_path = os.path.join(td, 'prefs.jsonl')
        build_preference_pairs(rollout_dir, pref_path, window_length=4)
        torch.save({'dummy': True}, os.path.join(td, 'init.ckpt'))
        import pickle
        with open(os.path.join(td, 'stats.pkl'), 'wb') as f:
            pickle.dump({'qpos_mean': np.zeros(14, dtype=np.float32), 'qpos_std': np.ones(14, dtype=np.float32), 'action_mean': np.zeros(14, dtype=np.float32), 'action_std': np.ones(14, dtype=np.float32)}, f)
        orig_build = dpo_mod.build_act_policy
        orig_load = dpo_mod.load_policy_checkpoint
        orig_traj = dpo_mod.trajectory_logprob
        try:
            dpo_mod.build_act_policy = lambda *_args, **_kwargs: _DummyPolicy()
            dpo_mod.load_policy_checkpoint = lambda *_args, **_kwargs: None
            dpo_mod.trajectory_logprob = lambda policy, *args, **kwargs: {
                'log_prob': policy.w.sum().unsqueeze(0),
                'entropy': torch.tensor([0.1]),
                'value': torch.tensor([0.0]),
            }
            from types import SimpleNamespace
            args = SimpleNamespace(
                task_name='sim_transfer_cube_scripted',
                pref_file=pref_path,
                dataset_stats=os.path.join(td, 'stats.pkl'),
                init_ckpt=os.path.join(td, 'init.ckpt'),
                reference_ckpt=None,
                out_dir=os.path.join(td, 'dpo_out'),
                device='cpu',
                max_steps=1,
                beta=0.1,
                grad_clip=1.0,
                checkpoint_interval=1,
                resume_ckpt='auto',
                temporal_agg=True,
                decay_k=0.01,
            )
            dpo_mod.train_dpo(args)
        finally:
            dpo_mod.build_act_policy = orig_build
            dpo_mod.load_policy_checkpoint = orig_load
            dpo_mod.trajectory_logprob = orig_traj
        assert os.path.isfile(os.path.join(td, 'dpo_out', 'policy_pref_last.ckpt'))
        assert os.path.isfile(os.path.join(td, 'dpo_out', 'policy_pref_best.ckpt'))
        assert os.path.isfile(os.path.join(td, 'dpo_out', 'dpo_state.pt'))
        assert os.path.isfile(os.path.join(td, 'dpo_out', 'metrics.jsonl'))


def test_ppo_one_update_smoke_writes_checkpoint_and_log():
    with tempfile.TemporaryDirectory() as td:
        torch.save({'dummy': True}, os.path.join(td, 'init.ckpt'))
        import pickle
        with open(os.path.join(td, 'stats.pkl'), 'wb') as f:
            pickle.dump({'qpos_mean': np.zeros(14, dtype=np.float32), 'qpos_std': np.ones(14, dtype=np.float32), 'action_mean': np.zeros(14, dtype=np.float32), 'action_std': np.ones(14, dtype=np.float32)}, f)
        orig_build = ppo_mod.build_act_policy
        orig_load = ppo_mod.load_policy_checkpoint
        orig_collect = ppo_mod.collect_policy_rollouts
        orig_eval = ppo_mod.evaluate_rollout_batch
        try:
            ppo_mod.build_act_policy = lambda *_args, **_kwargs: _DummyPolicy()
            ppo_mod.load_policy_checkpoint = lambda *_args, **_kwargs: None
            ppo_mod.collect_policy_rollouts = lambda *args, **kwargs: [{
                'images': [{'top': np.zeros((4, 4, 3), dtype=np.uint8)} for _ in range(2)],
                'qpos': np.zeros((2, 14), dtype=np.float32),
                'actions_norm': np.zeros((2, 14), dtype=np.float32),
                'rewards': np.array([0.0, 1.0], dtype=np.float32),
                'dones': np.array([0.0, 1.0], dtype=np.float32),
                'old_log_probs': np.zeros((2,), dtype=np.float32),
                'old_values': np.zeros((2,), dtype=np.float32),
                'returns': np.ones((2,), dtype=np.float32),
                'advantages': np.ones((2,), dtype=np.float32),
                'episode_return': 1.0,
                'highest_reward': 1.0,
                'success': True,
                'env_max_reward': 1.0,
            }]
            ppo_mod.evaluate_rollout_batch = lambda policy, *args, **kwargs: [{
                'new_log_probs': policy.w.repeat(2),
                'entropies': torch.ones(2),
                'values': policy.w.repeat(2),
                'old_log_probs': torch.zeros(2),
                'returns': torch.ones(2),
                'advantages': torch.ones(2),
            }]
            from types import SimpleNamespace
            args = SimpleNamespace(
                task_name='sim_transfer_cube_scripted',
                dataset_stats=os.path.join(td, 'stats.pkl'),
                init_ckpt=os.path.join(td, 'init.ckpt'),
                out_dir=os.path.join(td, 'ppo_out'),
                device='cpu',
                updates=1,
                num_rollouts=1,
                update_epochs=1,
                clip_eps=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                grad_clip=1.0,
                checkpoint_interval=1,
                resume_ckpt='auto',
                temporal_agg=True,
                decay_k=0.01,
                seed_start=3000,
                max_timesteps=None,
            )
            ppo_mod.train_ppo_like(args)
        finally:
            ppo_mod.build_act_policy = orig_build
            ppo_mod.load_policy_checkpoint = orig_load
            ppo_mod.collect_policy_rollouts = orig_collect
            ppo_mod.evaluate_rollout_batch = orig_eval
        assert os.path.isfile(os.path.join(td, 'ppo_out', 'policy_ppo_last.ckpt'))
        assert os.path.isfile(os.path.join(td, 'ppo_out', 'policy_ppo_best.ckpt'))
        assert os.path.isfile(os.path.join(td, 'ppo_out', 'ppo_state.pt'))
        assert os.path.isfile(os.path.join(td, 'ppo_out', 'metrics.jsonl'))


def main():
    test_rollout_artifact_round_trip_contains_required_metadata()
    test_preference_pairs_rank_success_before_return()
    test_rollout_schema_rejects_missing_required_arrays()
    test_rollout_schema_rejects_num_steps_mismatch()
    test_rollout_schema_rejects_action_shape_mismatch()
    test_dpo_rejects_preference_dataset_stats_mismatch()
    test_dpo_rejects_preference_source_checkpoint_mismatch()
    test_posttrain_policy_loads_legacy_bc_checkpoint_with_expected_missing_keys_only()
    test_dpo_one_step_smoke_writes_checkpoint_and_log()
    test_ppo_one_update_smoke_writes_checkpoint_and_log()
    print('All posttrain regression smoke tests passed.')


if __name__ == '__main__':
    main()
