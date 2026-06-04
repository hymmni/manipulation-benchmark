import math
import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Diffusion timestep embedding via sinusoidal encoding (like FourierFeatures in LDP)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) int or float -> (B, dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )
        x = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (B, dim)


class _MLPResNetBlock(nn.Module):
    def __init__(self, hidden_dim: int, use_layer_norm: bool):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return residual + x


# From: references/latent_diffusion_planning/networks/mlp_diffusion_nets.py
class MLPResNet(nn.Module):
    """MLP with residual blocks. PyTorch re-implementation of JAX/Flax MLPResNet."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, n_blocks: int = 2, use_layer_norm: bool = True):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            _MLPResNetBlock(hidden_dim, use_layer_norm) for _ in range(n_blocks)
        ])
        self.act = nn.ReLU()
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.act(x)
        return self.proj_out(x)


class ConditionalDenoiser(nn.Module):
    """noisy_x + cond + t -> predicted noise eps (same shape as x).
    Shared by LatentTrajectoryDiffusion and InverseDynamicsDiffusion."""

    def __init__(self, x_dim: int, cond_dim: int, hidden_dim: int, time_dim: int = 64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.net = MLPResNet(
            in_dim=x_dim + cond_dim + time_dim,
            out_dim=x_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x_noisy: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        # x_noisy: (B, x_dim), cond: (B, cond_dim), t: (B,)
        t_emb = self.time_emb(t)  # (B, time_dim)
        inp = torch.cat([x_noisy, cond, t_emb], dim=-1)
        return self.net(inp)
