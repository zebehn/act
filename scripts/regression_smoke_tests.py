#!/usr/bin/env python3
import os
import tempfile

import h5py
import numpy as np
import torch

import imitate_episodes
from utils import load_data


def _write_episode(path, value, is_sim=True):
    t = 2
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = is_sim
        obs = root.create_group("observations")
        image = obs.create_group("images")
        image.create_dataset("top", data=np.full((t, 4, 4, 3), value, dtype=np.uint8))
        obs.create_dataset("qpos", data=np.full((t, 14), value, dtype=np.float32))
        obs.create_dataset("qvel", data=np.full((t, 14), value, dtype=np.float32))
        root.create_dataset("action", data=np.full((t, 14), value, dtype=np.float32))


def test_non_contiguous_episode_ids():
    with tempfile.TemporaryDirectory() as td:
        _write_episode(os.path.join(td, "episode_0.hdf5"), value=0.0)
        _write_episode(os.path.join(td, "episode_1.hdf5"), value=100.0)  # should be ignored
        _write_episode(os.path.join(td, "episode_2.hdf5"), value=4.0)

        train_loader, val_loader, stats, _ = load_data(
            dataset_dir=td,
            num_episodes=2,
            camera_names=["top"],
            batch_size_train=1,
            batch_size_val=1,
            device=torch.device("cpu"),
            episode_ids=[0, 2],
        )

        # stats should be computed only from episodes 0 and 2 -> mean 2.0
        assert np.allclose(stats["action_mean"], np.full((14,), 2.0), atol=1e-6)
        assert np.allclose(stats["qpos_mean"], np.full((14,), 2.0), atol=1e-6)

        # ensure both loaders can produce at least one batch
        next(iter(train_loader))
        next(iter(val_loader))


class _DummyPolicy:
    def __init__(self):
        self._state = {"w": torch.tensor([1.0])}

    def to(self, _device):
        return self

    def train(self):
        return self

    def eval(self):
        return self

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state
        return None


class _DummyOptimizer:
    def zero_grad(self):
        return None

    def step(self):
        return None


def test_resume_when_start_epoch_exceeds_num_epochs():
    with tempfile.TemporaryDirectory() as td:
        resume_path = os.path.join(td, "policy_epoch_9_seed_0.ckpt")
        torch.save({"w": torch.tensor([3.0])}, resume_path)

        val_batch = (
            torch.zeros((1, 1, 3, 4, 4), dtype=torch.float32),
            torch.zeros((1, 14), dtype=torch.float32),
            torch.zeros((1, 2, 14), dtype=torch.float32),
            torch.zeros((1, 2), dtype=torch.bool),
        )
        val_loader = [val_batch]
        train_loader = [val_batch]

        orig_make_policy = imitate_episodes.make_policy
        orig_make_optimizer = imitate_episodes.make_optimizer
        orig_forward_pass = imitate_episodes.forward_pass
        orig_resolve_resume = imitate_episodes.resolve_resume_checkpoint
        try:
            imitate_episodes.make_policy = lambda *_args, **_kwargs: _DummyPolicy()
            imitate_episodes.make_optimizer = lambda *_args, **_kwargs: _DummyOptimizer()
            imitate_episodes.forward_pass = (
                lambda _data, _policy, _device: {"loss": torch.tensor(0.5, dtype=torch.float32, requires_grad=True)}
            )
            imitate_episodes.resolve_resume_checkpoint = lambda *_args, **_kwargs: resume_path

            cfg = {
                "num_epochs": 5,
                "ckpt_dir": td,
                "seed": 0,
                "policy_class": "ACT",
                "policy_config": {},
                "device": torch.device("cpu"),
                "resume_ckpt": "auto",
            }
            best_ckpt_info = imitate_episodes.train_bc(train_loader, val_loader, cfg)
        finally:
            imitate_episodes.make_policy = orig_make_policy
            imitate_episodes.make_optimizer = orig_make_optimizer
            imitate_episodes.forward_pass = orig_forward_pass
            imitate_episodes.resolve_resume_checkpoint = orig_resolve_resume

        best_epoch, best_loss, state = best_ckpt_info
        assert best_epoch == 9
        assert abs(best_loss - 0.5) < 1e-6
        assert "w" in state
        assert os.path.isfile(os.path.join(td, "policy_last.ckpt"))
        assert os.path.isfile(os.path.join(td, "policy_epoch_9_seed_0.ckpt"))


def main():
    test_non_contiguous_episode_ids()
    test_resume_when_start_epoch_exceeds_num_epochs()
    print("All regression smoke tests passed.")


if __name__ == "__main__":
    main()
