"""Shape + behavior tests for MVP neural network stubs. All tests run on CPU."""
import torch
import pytest

from config.config import Config, CVAEConfig, PlannerConfig, IDMConfig
from models.cvae import CVAE
from models.latent_traj_diffusion import LatentTrajectoryDiffusion
from models.inverse_dynamics import InverseDynamicsDiffusion

CFG = Config.default()
DEVICE = torch.device("cpu")

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _small_planner_cfg() -> PlannerConfig:
    cfg = PlannerConfig()
    cfg.n_diffusion_steps = 5
    cfg.hidden_dim = 32
    cfg.horizon = 4
    return cfg


def _small_idm_cfg() -> IDMConfig:
    cfg = IDMConfig()
    cfg.n_diffusion_steps = 5
    cfg.hidden_dim = 32
    cfg.action_horizon = 3
    return cfg


def _small_cvae_cfg() -> CVAEConfig:
    cfg = CVAEConfig()
    cfg.latent_dim = 8
    cfg.hidden_dim = 32
    return cfg


# ------------------------------------------------------------------ #
# CVAE                                                                 #
# ------------------------------------------------------------------ #

class TestCVAE:
    def setup_method(self):
        self.input_dim = 6
        self.cond_dim = 4
        self.B = 3
        self.model = CVAE(_small_cvae_cfg(), self.input_dim, self.cond_dim).to(DEVICE)

    def test_forward_shapes(self):
        x = torch.randn(self.B, self.input_dim)
        cond = torch.randn(self.B, self.cond_dim)
        recon, mu, logvar = self.model(x, cond)
        assert recon.shape == (self.B, self.input_dim)
        assert mu.shape == (self.B, _small_cvae_cfg().latent_dim)
        assert logvar.shape == (self.B, _small_cvae_cfg().latent_dim)

    def test_forward_no_cond(self):
        x = torch.randn(self.B, self.input_dim)
        model_no_cond = CVAE(_small_cvae_cfg(), self.input_dim, cond_dim=0).to(DEVICE)
        recon, mu, logvar = model_no_cond(x)
        assert recon.shape == (self.B, self.input_dim)

    def test_loss_is_scalar_dict(self):
        x = torch.randn(self.B, self.input_dim)
        cond = torch.randn(self.B, self.cond_dim)
        loss_dict = self.model.loss(x, cond)
        assert set(loss_dict.keys()) == {"recon", "kl", "total"}
        for v in loss_dict.values():
            assert v.shape == ()  # scalar


# ------------------------------------------------------------------ #
# LatentTrajectoryDiffusion                                            #
# ------------------------------------------------------------------ #

class TestLatentTrajectoryDiffusion:
    def setup_method(self):
        self.state_dim = 3
        self.cond_dim = 5
        self.cfg = _small_planner_cfg()
        self.model = LatentTrajectoryDiffusion(self.cfg, self.state_dim, self.cond_dim).to(DEVICE)

    def test_sample_shape(self):
        cond = torch.randn(1, self.cond_dim)
        out = self.model.sample(cond, num_samples=2)
        assert out.shape == (2, self.cfg.horizon, self.state_dim), f"Got {out.shape}"

    def test_loss_scalar(self):
        B = 4
        batch = {
            "traj": torch.randn(B, self.cfg.horizon, self.state_dim),
            "cond": torch.randn(B, self.cond_dim),
        }
        loss = self.model.loss(batch)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_cpu_only(self):
        for p in self.model.parameters():
            assert p.device.type == "cpu"


# ------------------------------------------------------------------ #
# InverseDynamicsDiffusion                                             #
# ------------------------------------------------------------------ #

class TestInverseDynamicsDiffusion:
    def setup_method(self):
        self.state_dim = 3
        self.action_dim = 3
        self.cfg = _small_idm_cfg()
        self.model = InverseDynamicsDiffusion(self.cfg, self.state_dim, self.action_dim).to(DEVICE)

    def test_sample_shape(self):
        cond = torch.randn(1, self.state_dim * 2)
        out = self.model.sample(cond, num_samples=4, exploration_noise=0.0)
        assert out.shape == (4, self.cfg.action_horizon, self.action_dim), f"Got {out.shape}"

    def test_loss_scalar(self):
        B = 4
        batch = {
            "actions": torch.randn(B, self.cfg.action_horizon, self.action_dim),
            "cond": torch.randn(B, self.state_dim * 2),
        }
        loss = self.model.loss(batch)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_exploration_noise_increases_diversity(self):
        """exploration_noise > 0 must produce more diverse samples than noise = 0."""
        cond = torch.zeros(1, self.state_dim * 2)

        torch.manual_seed(42)
        out_no_noise = self.model.sample(cond, num_samples=8, exploration_noise=0.0)
        std_no_noise = out_no_noise.std().item()

        torch.manual_seed(42)
        out_with_noise = self.model.sample(cond, num_samples=8, exploration_noise=0.5)
        std_with_noise = out_with_noise.std().item()

        assert std_with_noise > std_no_noise, (
            f"exploration_noise=0.5 std {std_with_noise:.4f} should exceed "
            f"exploration_noise=0.0 std {std_no_noise:.4f}"
        )

    def test_determinism_without_noise(self):
        """With fixed seed and no exploration_noise, two runs should match closely."""
        cond = torch.zeros(1, self.state_dim * 2)

        torch.manual_seed(0)
        out1 = self.model.sample(cond, num_samples=2, exploration_noise=0.0)

        torch.manual_seed(0)
        out2 = self.model.sample(cond, num_samples=2, exploration_noise=0.0)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_cpu_only(self):
        for p in self.model.parameters():
            assert p.device.type == "cpu"
