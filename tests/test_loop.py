"""Smoke tests for step 4: self-improve loop."""

import os
import tempfile

import numpy as np
import pytest

from config.config import Config
from data.replay_buffer import ReplayBuffer
from data.self_collected_buffer import SelfCollectedBuffer
from data.zarr_io import save_episode
from trainer.efficiency_filter import baseline_cost, is_efficient


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_store(tmp_path, n_episodes=3, ep_len=20):
    store = str(tmp_path / "test.zarr")
    rng = np.random.default_rng(42)
    for i in range(n_episodes):
        states = rng.random((ep_len, 3)).astype(np.float32)
        actions = rng.random((ep_len, 3)).astype(np.float32)
        meta = {"is_success": True, "path_len": float(100 + i * 10), "n_steps": ep_len}
        save_episode(store, states, actions, meta)
    return store


# ── ReplayBuffer ──────────────────────────────────────────────────────────────

def test_replay_buffer_len(tmp_path):
    store = _make_store(tmp_path, n_episodes=2, ep_len=10)
    buf = ReplayBuffer.from_zarr(store, horizon=4, action_horizon=2)
    # Each episode has 10 steps; window = max(4, 2) = 4; valid starts per ep = 10 - 4 + 1 = 7
    assert len(buf) == 7 * 2


def test_replay_buffer_sample_shapes(tmp_path):
    store = _make_store(tmp_path, n_episodes=2, ep_len=10)
    buf = ReplayBuffer.from_zarr(store, horizon=4, action_horizon=2)
    batch = buf.sample(8)
    assert batch["states"].shape == (8, 4, 3)
    assert batch["actions"].shape == (8, 2, 3)


def test_replay_buffer_no_cross_boundary(tmp_path):
    """Sampled windows must not cross episode boundaries."""
    store = str(tmp_path / "b.zarr")
    # 2 episodes of length 5 — total 10 timesteps
    for ep_i in range(2):
        states = np.full((5, 3), fill_value=ep_i, dtype=np.float32)
        save_episode(store, states)
    buf = ReplayBuffer.from_zarr(store, horizon=3, action_horizon=2)
    for _ in range(100):
        batch = buf.sample(4)
        # All states in a window must be from the same episode (all same value)
        for b in range(4):
            window = batch["states"][b]  # (3, 3)
            vals = window[:, 0]
            assert vals.min() == vals.max(), "Window crossed an episode boundary"


def test_replay_buffer_empty_no_crash(tmp_path):
    store = str(tmp_path / "empty.zarr")
    buf = ReplayBuffer(store, horizon=4, action_horizon=2)
    assert len(buf) == 0
    with pytest.raises(RuntimeError):
        buf.sample(1)


def test_self_collected_buffer_add(tmp_path):
    store = str(tmp_path / "self.zarr")
    buf = SelfCollectedBuffer(store, horizon=3, action_horizon=2)
    states = np.random.rand(10, 3).astype(np.float32)
    actions = np.random.rand(10, 3).astype(np.float32)
    meta = {"is_success": True, "path_len": 50.0, "had_collision": False}
    buf.add_episode(states, actions, meta)
    assert len(buf) > 0


# ── efficiency_filter ─────────────────────────────────────────────────────────

def test_baseline_cost_uses_path_len(tmp_path):
    store = _make_store(tmp_path, n_episodes=3, ep_len=10)
    buf = ReplayBuffer.from_zarr(store, horizon=4, action_horizon=2)
    eps = buf.episodes()
    cost = baseline_cost(eps)
    assert cost > 0


def test_is_efficient_shortcut_passes():
    ep_meta = {"is_success": True, "path_len": 50.0, "had_collision": False}
    assert is_efficient(ep_meta, baseline=100.0, margin=0.2)


def test_is_efficient_failure_rejected():
    ep_meta = {"is_success": False, "path_len": 30.0, "had_collision": False}
    assert not is_efficient(ep_meta, baseline=100.0, margin=0.2)


def test_is_efficient_collision_rejected():
    ep_meta = {"is_success": True, "path_len": 30.0, "had_collision": True}
    assert not is_efficient(ep_meta, baseline=100.0, margin=0.2)


def test_is_efficient_not_fast_enough():
    ep_meta = {"is_success": True, "path_len": 90.0, "had_collision": False}
    # threshold = 100 * (1 - 0.2) = 80 → 90 >= 80 → rejected
    assert not is_efficient(ep_meta, baseline=100.0, margin=0.2)


def test_is_efficient_with_discriminator():
    ep_meta = {"is_success": True, "path_len": 50.0, "had_collision": False}
    # discriminator vetoes the episode
    assert not is_efficient(ep_meta, 100.0, 0.2, discriminator=lambda m: False)


# ── dry-run integration ───────────────────────────────────────────────────────

def test_dry_run_completes():
    """Full pipeline dry-run: must complete 1 iteration without exception."""
    from main import main
    main(dry_run=True)
