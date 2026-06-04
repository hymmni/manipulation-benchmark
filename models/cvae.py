import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.config import CVAEConfig
from models.nets import MLPResNet


class CVAE(nn.Module):
    """Conditional Variational Autoencoder.
    Encodes trajectories/states into a latent representation for the planner."""

    def __init__(self, cfg: CVAEConfig, input_dim: int, cond_dim: int = 0):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        enc_in = input_dim + cond_dim
        self.encoder = MLPResNet(enc_in, cfg.hidden_dim, hidden_dim=cfg.hidden_dim)
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        dec_in = cfg.latent_dim + cond_dim
        self.decoder = MLPResNet(dec_in, input_dim, hidden_dim=cfg.hidden_dim)

    # Bound logvar so exp() cannot overflow to inf/NaN — robust even when inputs
    # are unnormalized (raw env coordinates up to ~900). Standard VAE practice.
    LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0

    def encode(self, x: Tensor, cond: Tensor | None = None) -> tuple[Tensor, Tensor]:
        inp = torch.cat([x, cond], dim=-1) if cond is not None else x
        h = self.encoder(inp)
        logvar = self.fc_logvar(h).clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
        return self.fc_mu(h), logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, cond: Tensor | None = None) -> Tensor:
        inp = torch.cat([z, cond], dim=-1) if cond is not None else z
        return self.decoder(inp)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

    def loss(self, x: Tensor, cond: Tensor | None = None, beta: float = 1.0) -> dict:
        recon, mu, logvar = self.forward(x, cond)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + beta * kl_loss
        return {"recon": recon_loss, "kl": kl_loss, "total": total}
