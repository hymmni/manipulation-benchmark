import torch
import torch.nn.functional as F
from torch import Tensor

from config.config import IDMConfig
from models.base import BaseGenerativeModel
from models.nets import ConditionalDenoiser


class InverseDynamicsDiffusion(BaseGenerativeModel):
    """Diffusion model for inverse dynamics: given adjacent states, generate the bridging actions.
    exploration_noise injects additional Gaussian noise into each reverse-diffusion step
    (on-manifold exploration, DPPO-aligned) — not post-hoc noise on the final action."""

    def __init__(self, cfg: IDMConfig, state_dim: int, action_dim: int):
        super().__init__(backbone=cfg.backbone)
        self.cfg = cfg
        self.action_dim = action_dim
        self.action_horizon = cfg.action_horizon
        self.x_dim = action_dim * cfg.action_horizon  # flattened action sequence

        # cond = (s_t, s_{t+1}) concatenated
        cond_dim = state_dim * 2
        self.denoiser = ConditionalDenoiser(
            x_dim=self.x_dim,
            cond_dim=cond_dim,
            hidden_dim=cfg.hidden_dim,
        )
        self._n_steps = cfg.n_diffusion_steps

    def loss(self, batch: dict) -> Tensor:
        """batch: 'actions' (B, action_horizon, action_dim), 'cond' (B, state_dim*2)."""
        actions = batch["actions"]
        cond = batch["cond"]
        B = actions.shape[0]
        device = actions.device

        x0 = actions.reshape(B, -1)
        noise = torch.randn_like(x0)
        t = torch.randint(0, self._n_steps, (B,), device=device)
        noisy = self.scheduler.add_noise(x0, noise, t)

        eps_pred = self.denoiser(noisy, cond, t)
        return F.mse_loss(eps_pred, noise)

    def sample(
        self,
        cond: Tensor,
        *,
        num_samples: int = 1,
        exploration_noise: float = 0.0,
    ) -> Tensor:
        """Return (num_samples, action_horizon, action_dim).
        cond: (1, state_dim*2) — pair of adjacent states.
        exploration_noise > 0: injects additional Gaussian noise at each reverse-diffusion step
        to encourage diversity (on-manifold exploration). exploration_noise = 0 is near-deterministic
        under fixed seed."""
        device = cond.device
        cond_rep = cond.expand(num_samples, -1)  # (ns, cond_dim)

        self.scheduler.set_timesteps(self._n_steps)
        x = torch.randn(num_samples, self.x_dim, device=device)

        for t_val in self.scheduler.timesteps:
            t_batch = torch.full((num_samples,), t_val, device=device, dtype=torch.long)
            eps_pred = self.denoiser(x, cond_rep, t_batch)
            step_out = self.scheduler.step(eps_pred, t_val, x)
            x = step_out.prev_sample

            # Inject exploration noise into posterior sample at each reverse step
            if exploration_noise > 0.0:
                x = x + exploration_noise * torch.randn_like(x)

        return x.reshape(num_samples, self.action_horizon, self.action_dim)
