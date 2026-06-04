import math
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless guard before any pg import

from env.svg_map import load_map
from env.trajectory_explore_env import TrajectoryExploreEnv


# ── load_map ──────────────────────────────────────────────────────────────────

def test_load_map_obstacle_count():
    m = load_map("references/image.svg")
    assert len(m.obstacles) >= 15


def test_load_map_agent_start():
    m = load_map("references/image.svg")
    x, y, yaw = m.agent_start
    assert abs(x - 130.0) < 1.0
    assert abs(y - 300.0) < 1.0
    assert abs(yaw) < 0.01  # ~0 radians


def test_load_map_goal_center():
    m = load_map("references/image.svg")
    gx, gy = m.goal_center
    assert abs(gx - 800.0) < 5.0
    assert abs(gy - 300.0) < 5.0


# ── env interface ─────────────────────────────────────────────────────────────

def _make_env():
    return TrajectoryExploreEnv(render_mode=None)


def test_reset_returns_obs_and_info():
    env = _make_env()
    result = env.reset(seed=0)
    assert isinstance(result, tuple) and len(result) == 2
    obs, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    env.close()


def test_reset_seed_reproducibility():
    env = _make_env()
    obs1, _ = env.reset(seed=0)
    obs2, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs1, obs2)
    env.close()


def test_obs_shape_dtype():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    env.close()


def test_obs_in_space():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), f"obs {obs} not in space"
    env.close()


def test_step_returns_five_tuple():
    env = _make_env()
    env.reset(seed=0)
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple) and len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_step_info_keys():
    env = _make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "is_success" in info
    assert "path_len" in info
    assert "collision" in info
    env.close()


def test_step_obs_in_space():
    env = _make_env()
    env.reset(seed=0)
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
    env.close()


# ── render rgb_array ─────────────────────────────────────────────────────────

def test_render_rgb_array_shape():
    env = TrajectoryExploreEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    # height × width × 3
    assert frame.shape == (env.cfg.height, env.cfg.width, 3)
    env.close()


def test_render_rgb_array_after_step():
    env = TrajectoryExploreEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    frame = env.render()
    assert frame is not None and frame.dtype == np.uint8
    env.close()


# ── gymnasium env_checker ─────────────────────────────────────────────────────

def test_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env
    env = TrajectoryExploreEnv(render_mode=None)
    check_env(env, skip_render_check=True)
    env.close()
