import torch
import torch.nn.functional as F
from torch import Tensor

from config.config import PlannerConfig
from models.base import BaseGenerativeModel
from models.nets import ConditionalDenoiser


class LatentTrajectoryDiffusion(BaseGenerativeModel):
    """Goal-conditioned diffusion planner over future state trajectories.
    Supports classifier-free guidance (CFG) for goal conditioning."""

    def __init__(self, cfg: PlannerConfig, state_dim: int, cond_dim: int):
        super().__init__(backbone=cfg.backbone)
        self.cfg = cfg
        self.state_dim = state_dim
        self.horizon = cfg.horizon
        self.cond_dropout_prob = cfg.cond_dropout_prob
        self.x_dim = state_dim * cfg.horizon  # flattened trajectory

        self.register_buffer("null_cond", torch.zeros(1, cond_dim))

        self.denoiser = ConditionalDenoiser(
            x_dim=self.x_dim,
            cond_dim=cond_dim,
            hidden_dim=cfg.hidden_dim,
        )
        self._n_steps = cfg.n_diffusion_steps

    def loss(self, batch: dict) -> Tensor:
        """batch: 'traj' (B, horizon, state_dim), 'cond' (B, cond_dim)."""
        traj = batch["traj"]
        cond = batch["cond"]
        B = traj.shape[0]
        device = traj.device

        x0 = traj.reshape(B, -1)
        noise = torch.randn_like(x0)
        t = torch.randint(0, self._n_steps, (B,), device=device)
        noisy = self.scheduler.add_noise(x0, noise, t)

        if self.cond_dropout_prob > 0.0:
            drop_mask = torch.rand(B, device=device) < self.cond_dropout_prob
            null = self.null_cond.expand(B, -1)
            cond = torch.where(drop_mask.unsqueeze(-1), null, cond)

        eps_pred = self.denoiser(noisy, cond, t)
        return F.mse_loss(eps_pred, noise)

    def sample(
        self,
        cond: Tensor,
        *,
        num_samples: int = 1,
        guidance_scale: float | None = None,
    ) -> Tensor:
        """Return (num_samples, horizon, state_dim).
        cond: (1, cond_dim) — single condition vector."""
        scale = guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        device = cond.device

        # Tile condition for each sample
        cond_rep = cond.expand(num_samples, -1)  # (ns, cond_dim)
        null_rep = self.null_cond.expand(num_samples, -1)

        self.scheduler.set_timesteps(self._n_steps)
        x = torch.randn(num_samples, self.x_dim, device=device)

        for t_val in self.scheduler.timesteps:
            t_batch = torch.full((num_samples,), t_val, device=device, dtype=torch.long)
            eps_cond = self.denoiser(x, cond_rep, t_batch)

            if scale > 0.0:
                eps_uncond = self.denoiser(x, null_rep, t_batch)
                eps = eps_uncond + scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            x = self.scheduler.step(eps, t_val, x).prev_sample

        return x.reshape(num_samples, self.horizon, self.state_dim)
