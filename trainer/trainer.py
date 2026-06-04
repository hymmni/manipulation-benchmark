import numpy as np
import torch
import torch.nn as nn

from config.config import Config
from models.cvae import CVAE
from models.latent_traj_diffusion import LatentTrajectoryDiffusion
from models.inverse_dynamics import InverseDynamicsDiffusion
from utils.device import get_device


class Trainer:
    """Orchestrates CVAE / Planner / IDM training and rollout for the self-improve loop."""

    # obs = (x, y, yaw, goal_dx, goal_dy) — 5D
    _OBS_DIM = 5

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.planner: LatentTrajectoryDiffusion | None = None
        self.idm: InverseDynamicsDiffusion | None = None
        self.cvae: CVAE | None = None
        self._planner_opt: torch.optim.Optimizer | None = None
        self._idm_opt: torch.optim.Optimizer | None = None
        self._cvae_opt: torch.optim.Optimizer | None = None

    # ── model construction ────────────────────────────────────────────────────

    def build_models(self) -> None:
        m = self.cfg.model
        state_dim = m.state_dim
        action_dim = m.action_dim
        obs_dim = self._OBS_DIM  # planner conditioning: full 5D observation

        self.cvae = CVAE(
            m.cvae,
            input_dim=state_dim * m.planner.horizon,
            cond_dim=obs_dim,
        ).to(self.device)

        self.planner = LatentTrajectoryDiffusion(
            m.planner,
            state_dim=state_dim,
            cond_dim=obs_dim,
        ).to(self.device)

        # IDM cond = concat(s_t, s_{t+action_horizon}) → state_dim * 2
        self.idm = InverseDynamicsDiffusion(
            m.idm,
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)

        lr = self.cfg.train.lr
        self._cvae_opt = torch.optim.Adam(self.cvae.parameters(), lr=lr)
        self._planner_opt = torch.optim.Adam(self.planner.parameters(), lr=lr)
        self._idm_opt = torch.optim.Adam(self.idm.parameters(), lr=lr)

    # ── batch helpers ─────────────────────────────────────────────────────────

    def _planner_batch(self, raw: dict) -> dict:
        states = raw["states"].to(self.device)  # (B, H, 3)
        first = states[:, 0, :]  # (B, 3)
        gx, gy = self.cfg.env.goal_center
        goal_dx = gx - first[:, 0:1]
        goal_dy = gy - first[:, 1:2]
        cond = torch.cat([first, goal_dx, goal_dy], dim=-1)  # (B, 5)
        return {"traj": states, "cond": cond}

    def _idm_batch(self, raw: dict) -> dict:
        states = raw["states"].to(self.device)   # (B, H, 3)
        actions = raw["actions"].to(self.device)  # (B, AH, 3)
        ah = self.cfg.model.idm.action_horizon
        s_t = states[:, 0, :]
        s_t_ah = states[:, min(ah, states.shape[1] - 1), :]
        cond = torch.cat([s_t, s_t_ah], dim=-1)  # (B, 6)
        return {"actions": actions, "cond": cond}

    def _cvae_batch(self, raw: dict) -> dict:
        states = raw["states"].to(self.device)   # (B, H, 3)
        B, H, D = states.shape
        first = states[:, 0, :]
        gx, gy = self.cfg.env.goal_center
        goal_dx = gx - first[:, 0:1]
        goal_dy = gy - first[:, 1:2]
        cond = torch.cat([first, goal_dx, goal_dy], dim=-1)
        x = states.reshape(B, H * D)
        return {"x": x, "cond": cond}

    def _train_step(self, raw: dict) -> dict:
        losses = {}

        # CVAE
        cvae_b = self._cvae_batch(raw)
        cvae_loss_dict = self.cvae.loss(cvae_b["x"], cvae_b["cond"])
        self._cvae_opt.zero_grad()
        cvae_loss_dict["total"].backward()
        self._cvae_opt.step()
        losses["cvae"] = cvae_loss_dict["total"].item()

        # Planner
        plan_b = self._planner_batch(raw)
        plan_loss = self.planner.loss(plan_b)
        self._planner_opt.zero_grad()
        plan_loss.backward()
        self._planner_opt.step()
        losses["planner"] = plan_loss.item()

        # IDM
        idm_b = self._idm_batch(raw)
        idm_loss = self.idm.loss(idm_b)
        self._idm_opt.zero_grad()
        idm_loss.backward()
        self._idm_opt.step()
        losses["idm"] = idm_loss.item()

        return losses

    # ── training phases ───────────────────────────────────────────────────────

    def pretrain(self, demo_buffer, steps: int) -> dict:
        """Behavior cloning on human demos to warm-start all models."""
        if len(demo_buffer) == 0:
            return {}
        last_losses: dict = {}
        for _ in range(steps):
            raw = demo_buffer.sample(self.cfg.train.batch_size)
            last_losses = self._train_step(raw)
        return last_losses

    def finetune(self, mixed_buffers, steps: int) -> dict:
        # DPPO (Diffusion Policy Policy Optimization, ICLR'25) is a stronger upgrade path —
        # treats the reverse diffusion chain as an MDP and applies policy gradient finetuning.
        # MVP uses behavior cloning (diffusion loss) on shortcut episodes instead.
        valid = [b for b in mixed_buffers if len(b) > 0]
        if not valid:
            return {}
        last_losses: dict = {}
        for _ in range(steps):
            n_each = max(1, self.cfg.train.batch_size // len(valid))
            raw_batches = [b.sample(n_each) for b in valid]
            raw = {
                k: torch.cat([rb[k] for rb in raw_batches], dim=0)
                for k in raw_batches[0]
            }
            last_losses = self._train_step(raw)
        return last_losses

    # ── rollout ───────────────────────────────────────────────────────────────

    def rollout(
        self,
        env,
        exploration_noise: float,
        n_episodes: int,
    ) -> list[dict]:
        """Collect episodes using Planner + IDM with on-manifold exploration noise.

        1. Planner generates a future state trajectory (goal-conditioned, CFG).
        2. IDM bridges adjacent planned state pairs into actions with exploration_noise,
           encouraging discovery of trajectories absent from the demo dataset.
        3. Actual (not planned) states/actions are recorded for the buffer.
        """
        self.planner.eval()
        self.idm.eval()
        episodes = []
        ah = self.cfg.model.idm.action_horizon
        horizon = self.cfg.model.planner.horizon
        gx, gy = self.cfg.env.goal_center

        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = obs[:3].copy()  # (x, y, yaw)

            # Plan a goal-conditioned trajectory
            cond_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                planned = self.planner.sample(
                    cond_t,
                    num_samples=1,
                    guidance_scale=self.cfg.model.planner.guidance_scale,
                )[0]  # (H, 3)

            all_states: list[np.ndarray] = [state]
            all_actions: list[np.ndarray] = []
            ep_collision = False
            terminated = truncated = False
            info: dict = {"is_success": False, "path_len": 0.0, "collision": False}

            # Execute IDM actions for each adjacent planned-state pair
            for t in range(horizon - 1):
                if terminated or truncated:
                    break
                s_t = planned[t]
                s_t_ah = planned[min(t + ah, horizon - 1)]
                idm_cond = torch.cat([s_t, s_t_ah]).unsqueeze(0)  # (1, 6)

                with torch.no_grad():
                    actions_tensor = self.idm.sample(
                        idm_cond,
                        num_samples=1,
                        exploration_noise=exploration_noise,
                    )[0]  # (AH, 3)

                for a_i in range(ah):
                    action = actions_tensor[a_i].cpu().numpy()
                    obs, _, terminated, truncated, info = env.step(action)
                    all_states.append(obs[:3].copy())
                    all_actions.append(action.copy())
                    if info["collision"]:
                        ep_collision = True
                    if terminated or truncated:
                        break

            # Align: states[i] precedes actions[i]; drop trailing state
            n = len(all_actions)
            states_arr = np.array(all_states[:n], dtype=np.float32)
            actions_arr = np.array(all_actions, dtype=np.float32)

            ep_meta = {
                "is_success": bool(info.get("is_success", False)),
                "path_len": float(info.get("path_len", 0.0)),
                "n_steps": n,
                "had_collision": ep_collision,
            }
            episodes.append({"states": states_arr, "actions": actions_arr, "meta": ep_meta})

        self.planner.train()
        self.idm.train()
        return episodes

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, env, n_episodes: int) -> dict:
        eps = self.rollout(env, exploration_noise=0.0, n_episodes=n_episodes)
        success_rate = float(np.mean([ep["meta"]["is_success"] for ep in eps]))
        successful = [ep["meta"]["path_len"] for ep in eps if ep["meta"]["is_success"]]
        mean_cost = float(np.mean(successful)) if successful else float("nan")
        return {"success_rate": success_rate, "mean_cost": mean_cost}
