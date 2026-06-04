import os
import tempfile

import numpy as np
import pytest

from data.spline import interpolate_waypoints
from data.zarr_io import save_episode, load_episodes, num_episodes

# Waypoints that go up and around (biased above-center path)
_WAYPOINTS = np.array(
    [[130, 300], [300, 120], [500, 140], [700, 280], [800, 300]], dtype=float
)


def test_shape_and_dtype():
    traj = interpolate_waypoints(_WAYPOINTS, n_points=150)
    assert traj.shape == (150, 3)
    assert traj.dtype == np.float32


def test_yaw_continuity():
    traj = interpolate_waypoints(_WAYPOINTS, n_points=200)
    max_jump = float(np.abs(np.diff(traj[:, 2])).max())
    assert max_jump < np.pi, f"yaw jump {max_jump:.4f} >= π — unwrap failed"


def test_endpoints_close_to_waypoints():
    traj = interpolate_waypoints(_WAYPOINTS, n_points=100)
    start = _WAYPOINTS[0]
    end = _WAYPOINTS[-1]
    assert np.linalg.norm(traj[0, :2] - start) < 1.0
    assert np.linalg.norm(traj[-1, :2] - end) < 1.0


def test_two_waypoints_linear():
    wp = np.array([[0.0, 0.0], [100.0, 0.0]])
    traj = interpolate_waypoints(wp, n_points=50)
    assert traj.shape == (50, 3)
    assert traj.dtype == np.float32


def test_single_waypoint_raises():
    with pytest.raises(ValueError):
        interpolate_waypoints(np.array([[100.0, 200.0]]))


def test_zarr_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test.zarr")
        states = np.random.rand(50, 3).astype(np.float32)
        save_episode(store_path, states)
        data = load_episodes(store_path)
        np.testing.assert_array_almost_equal(data["states"], states)
        assert data["episode_ends"][0] == 50


def test_zarr_two_episode_append():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test.zarr")
        s1 = np.random.rand(30, 3).astype(np.float32)
        s2 = np.random.rand(40, 3).astype(np.float32)
        save_episode(store_path, s1)
        save_episode(store_path, s2)

        assert num_episodes(store_path) == 2
        data = load_episodes(store_path)
        assert data["episode_ends"][0] == 30
        assert data["episode_ends"][1] == 70
        np.testing.assert_array_almost_equal(data["states"][:30], s1)
        np.testing.assert_array_almost_equal(data["states"][30:], s2)


def test_zarr_meta():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test.zarr")
        states = np.ones((20, 3), dtype=np.float32)
        save_episode(store_path, states, meta={"is_demo": True, "path_len": 3.14})
        data = load_episodes(store_path)
        assert "is_demo" in data["meta"]
