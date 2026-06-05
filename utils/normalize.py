"""State / observation normalization to ~[-1, 1].

The planner and CVAE diffuse over *states*; with raw pixel coordinates (x∈[0,900],
y∈[0,600]) the DDPM noise-prediction target (N(0,1)) is swamped by the input scale
and the model cannot learn. Normalizing states to ~[-1,1] puts the diffusion target
on the same scale as the noise. The env stays in raw coordinates — normalization is
applied only at the model boundary (trainer batches + rollout conditioning).
"""

import math

import torch
from torch import Tensor


def normalize_state(s: Tensor, width: float, height: float) -> Tensor:
    """(..., 3) raw (x, y, yaw) → ~[-1, 1]. x by width, y by height, yaw by π."""
    x = s[..., 0:1] / width * 2.0 - 1.0
    y = s[..., 1:2] / height * 2.0 - 1.0
    yaw = s[..., 2:3] / math.pi
    return torch.cat([x, y, yaw], dim=-1)


def unnormalize_state(s: Tensor, width: float, height: float) -> Tensor:
    """Inverse of normalize_state."""
    x = (s[..., 0:1] + 1.0) * 0.5 * width
    y = (s[..., 1:2] + 1.0) * 0.5 * height
    yaw = s[..., 2:3] * math.pi
    return torch.cat([x, y, yaw], dim=-1)


def normalize_obs(obs: Tensor, width: float, height: float) -> Tensor:
    """(..., 5) = state(3) + goal_delta(2). State normalized; goal deltas scaled by map size."""
    s = normalize_state(obs[..., :3], width, height)
    gdx = obs[..., 3:4] / width
    gdy = obs[..., 4:5] / height
    return torch.cat([s, gdx, gdy], dim=-1)
