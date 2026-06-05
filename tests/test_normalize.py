import math

import torch

from utils.normalize import normalize_state, unnormalize_state, normalize_obs


W, H = 900.0, 600.0


def test_state_roundtrip():
    s = torch.tensor([[130.0, 300.0, 0.0], [800.0, 300.0, math.pi / 2]])
    back = unnormalize_state(normalize_state(s, W, H), W, H)
    assert torch.allclose(s, back, atol=1e-4)


def test_state_range():
    # Corners of the map map into [-1, 1]
    s = torch.tensor([[0.0, 0.0, -math.pi], [W, H, math.pi]])
    n = normalize_state(s, W, H)
    assert torch.allclose(n[0], torch.tensor([-1.0, -1.0, -1.0]), atol=1e-5)
    assert torch.allclose(n[1], torch.tensor([1.0, 1.0, 1.0]), atol=1e-5)


def test_normalize_obs_shape_and_state_part():
    obs = torch.tensor([[130.0, 300.0, 0.0, 670.0, 0.0]])  # state(3) + goal_delta(2)
    n = normalize_obs(obs, W, H)
    assert n.shape == (1, 5)
    # state part matches normalize_state
    assert torch.allclose(n[:, :3], normalize_state(obs[:, :3], W, H))
    # goal delta scaled by map size
    assert torch.allclose(n[:, 3:5], torch.tensor([[670.0 / W, 0.0]]))


def test_batched_trajectory():
    traj = torch.rand(4, 16, 3) * torch.tensor([W, H, 2 * math.pi]) - torch.tensor([0.0, 0.0, math.pi])
    n = normalize_state(traj, W, H)
    assert n.shape == traj.shape
    assert torch.allclose(unnormalize_state(n, W, H), traj, atol=1e-3)
