"""Self-improvement online loop for the manipulation-benchmark MVP.

Pipeline:
  1. Pretrain CVAE / Planner / IDM on human demos (behavior cloning).
  2. Self-grow loop (AdaptDiffuser-style, but with real env rollout):
     (a) Explore — IDM uses exploration_noise to discover novel paths.
     (b) Filter  — keep only goal-reaching, collision-free shortcuts
                   that beat the demo average cost.
     (c) Buffer  — accumulate shortcuts in SelfCollectedBuffer.
     (d) Finetune — mixed replay (demo + self) behaviour cloning.
     (e) Evaluate — track success_rate / mean_cost across iterations.

Run locally (CPU, no GPU required):
    python main.py --dry-run

Full training is meant to run on the GPU server (see CLAUDE.md, 3-PC Workflow).
"""

import argparse
import os
import sys
import tempfile

import numpy as np

from config.config import Config
from data.replay_buffer import ReplayBuffer
from data.self_collected_buffer import SelfCollectedBuffer
from data.spline import interpolate_waypoints
from data.zarr_io import save_episode
from env.trajectory_explore_env import TrajectoryExploreEnv
from trainer.efficiency_filter import baseline_cost, is_efficient
from trainer.trainer import Trainer
from utils.device import get_device


# ── dry-run helpers ───────────────────────────────────────────────────────────

def shrink_for_dry_run(cfg: Config) -> Config:
    """Shrink all dimensions so the full loop finishes in seconds on CPU."""
    cfg.model.planner.horizon = 3
    cfg.model.planner.n_diffusion_steps = 2
    cfg.model.planner.hidden_dim = 16
    cfg.model.idm.action_horizon = 2
    cfg.model.idm.n_diffusion_steps = 2
    cfg.model.idm.hidden_dim = 16
    cfg.model.cvae.hidden_dim = 16
    cfg.model.cvae.latent_dim = 4
    cfg.train.batch_size = 2
    cfg.train.pretrain_steps = 2
    cfg.train.finetune_steps = 2
    cfg.train.use_wandb = False
    cfg.env.max_steps = 30
    return cfg


def _make_synthetic_demos(store_path: str, n_demos: int = 3, n_points: int = 10) -> None:
    """Synthesize inefficient curved demos using spline interpolation."""
    rng = np.random.default_rng(0)
    start = np.array([130.0, 300.0])
    goal_xy = np.array([800.0, 300.0])

    for i in range(n_demos):
        # Detoured path — deliberately long so shortcuts can beat it
        mid = (start + goal_xy) / 2 + np.array([0.0, rng.uniform(-120, 120)])
        waypoints = np.stack([start, mid, goal_xy])
        traj = interpolate_waypoints(waypoints, n_points=n_points)  # (n_points, 3)

        # Synthetic actions: xy-displacement between consecutive states
        actions = np.zeros((n_points, 3), dtype=np.float32)
        actions[:-1, :2] = np.diff(traj[:, :2], axis=0) / 10.0  # normalise by action_scale

        path_len = float(np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)))
        meta = {
            "is_success": True,
            "path_len": path_len,
            "n_steps": n_points,
        }
        save_episode(store_path, traj, actions, meta)


# ── logging ───────────────────────────────────────────────────────────────────

def _log(metrics: dict, iteration: int) -> None:
    parts = [f"iter={iteration}"] + [f"{k}={v:.4g}" for k, v in metrics.items()]
    print("[loop]", " | ".join(parts), flush=True)


# ── main loop ─────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    cfg = Config.default()

    tmp_dir = None
    if dry_run:
        cfg = shrink_for_dry_run(cfg)
        tmp_dir = tempfile.mkdtemp(prefix="mbench_dryrun_")
        cfg.buffer.demo_path = os.path.join(tmp_dir, "demos.zarr")
        cfg.buffer.self_path = os.path.join(tmp_dir, "self.zarr")
        print(f"[dry-run] temp store: {tmp_dir}")

    try:
        _run(cfg, dry_run=dry_run)
    finally:
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _run(cfg: Config, dry_run: bool) -> None:
    device = get_device("cpu" if dry_run else cfg.train.device)
    print(f"[loop] device={device}")

    # ── environment ───────────────────────────────────────────────────────────
    env = TrajectoryExploreEnv(cfg.env, render_mode="rgb_array")

    # ── demo buffer ───────────────────────────────────────────────────────────
    horizon = cfg.model.planner.horizon
    action_horizon = cfg.model.idm.action_horizon

    if dry_run:
        n_pts = horizon + action_horizon + 2  # episodes long enough for at least one window
        _make_synthetic_demos(cfg.buffer.demo_path, n_demos=3, n_points=n_pts)
        print(f"[dry-run] synthesised 3 demo episodes ({n_pts} steps each)")

    demo_buf = ReplayBuffer.from_zarr(cfg.buffer.demo_path, horizon, action_horizon)
    self_buf = SelfCollectedBuffer(cfg.buffer.self_path, horizon, action_horizon)

    print(f"[loop] demo_buf windows={len(demo_buf)}, self_buf windows={len(self_buf)}")

    # ── models & optimisers ───────────────────────────────────────────────────
    trainer = Trainer(cfg, device)
    trainer.build_models()
    print("[loop] models built")

    # ── 1) Pretrain on (inefficient) human demos ──────────────────────────────
    pretrain_losses = trainer.pretrain(demo_buf, steps=cfg.train.pretrain_steps)
    print(f"[loop] pretrain done: {pretrain_losses}")

    base = baseline_cost(demo_buf.episodes())
    print(f"[loop] demo baseline cost = {base:.2f}")

    # ── 2) Self-growth loop ───────────────────────────────────────────────────
    n_iters = 1 if dry_run else 10
    n_rollout_eps = 2 if dry_run else 8
    n_eval_eps = 2 if dry_run else 5

    for it in range(n_iters):
        # (a) Exploration rollout: IDM uses exploration_noise to discover novel paths.
        #     Planner provides goal-conditioned waypoints (CFG guidance);
        #     IDM bridges adjacent waypoints with stochastic actions.
        episodes = trainer.rollout(
            env,
            exploration_noise=cfg.model.idm.exploration_noise,
            n_episodes=n_rollout_eps,
        )

        # (b) Efficiency filter: keep only goal-reaching, collision-free shortcuts
        #     that beat the demo average cost by the configured margin.
        shortcuts = [ep for ep in episodes if is_efficient(ep["meta"], base, cfg.buffer.efficiency_margin)]
        print(
            f"[loop] iter={it} | rolled={len(episodes)} | shortcuts={len(shortcuts)} "
            f"| success={sum(e['meta']['is_success'] for e in episodes)}"
        )

        # (c) Accumulate shortcuts in the self buffer
        for ep in shortcuts:
            if len(ep["states"]) > 0 and len(ep["actions"]) > 0:
                self_buf.add_episode(ep["states"], ep["actions"], ep["meta"])

        # (d) Mixed finetune: demo (human, inefficient) + self (shortcut, efficient)
        finetune_losses = trainer.finetune([demo_buf, self_buf], steps=cfg.train.finetune_steps)

        # (e) Evaluate: track whether success_rate / mean_cost self-improve
        metrics = trainer.evaluate(env, n_episodes=n_eval_eps)
        metrics["shortcuts_total"] = len(self_buf)
        metrics["finetune_loss_planner"] = finetune_losses.get("planner", float("nan"))
        _log(metrics, it)

        # WandB logging — only when explicitly enabled (never in dry-run / CI)
        if cfg.train.use_wandb:
            import wandb
            wandb.log({"iter": it, **metrics})

    env.close()
    print("[loop] done")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-improvement loop — manipulation-benchmark")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 iteration on CPU with tiny dims (no GPU, no wandb). "
             "Used to verify pipeline wiring before GPU training.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
