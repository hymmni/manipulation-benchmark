from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseGenerativeModel(nn.Module, ABC):
    """Common training/sampling skeleton wrapping a generative backbone (currently DDPM).
    Isolates backbone swap (flow_matching/shortcut) to _build_scheduler."""

    def __init__(self, backbone: str = "ddpm"):
        super().__init__()
        self.backbone = backbone
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self):
        if self.backbone == "ddpm":
            from diffusers import DDPMScheduler
            return DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
        if self.backbone in ("flow_matching", "shortcut"):
            raise NotImplementedError(f"backbone='{self.backbone}' is reserved for future use")
        raise ValueError(f"Unknown backbone: {self.backbone!r}")

    @abstractmethod
    def loss(self, batch: dict) -> Tensor:
        """Compute noise-prediction MSE loss. Returns scalar tensor."""

    @abstractmethod
    def sample(self, cond: Tensor, *, num_samples: int = 1) -> Tensor:
        """Draw samples conditioned on cond."""
