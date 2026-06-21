"""Human teleoperation demo collection — PushT-style mouse following.

The agent follows the mouse cursor (PushT convention): each step it moves toward the
cursor by at most ``action_scale`` pixels, and its heading (yaw) auto-aligns to the
direction of motion. Episodes are recorded into the same Zarr schema the training
pipeline consumes (real actions, not zero-filled), so no model code changes are needed.

Run on a machine WITH a display (the coding/local PC). Do NOT set SDL_VIDEODRIVER=dummy.
Transfer the resulting ``data_store/demos.zarr`` to the GPU server for training.

Controls
  mouse      drag the red agent toward the blue goal (move cursor; agent follows)
  R          discard the current episode and restart it
  N          save the current episode now (even if the goal isn't reached)
  ESC / quit stop collecting (already-saved episodes are kept)

Usage
  python collect_demos.py --demos 8
  python collect_demos.py --demos 8 --out data_store/demos.zarr --overwrite
"""

import argparse
import math
import os
import shutil
import sys

import numpy as np

from config.config import Config
from data.zarr_io import save_episode, num_episodes
from env.trajectory_explore_env import TrajectoryExploreEnv


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _mouse_action(agent_xy, agent_yaw, mouse_xy, scale: float) -> np.ndarray:
    """PushT-style follow: move toward cursor, auto-align yaw to motion direction.

    action[:2] = clip((cursor - agent) / scale, -1, 1)   — velocity toward the cursor
    action[2]  = yaw increment that rotates toward atan2(dy, dx), normalised to [-1, 1]
                 (env applies dyaw = action[2] * scale * 0.1). Held at 0 when ~stationary.
    """
    tx = mouse_xy[0] - agent_xy[0]
    ty = mouse_xy[1] - agent_xy[1]
    ax = float(np.clip(tx / scale, -1.0, 1.0))
    ay = float(np.clip(ty / scale, -1.0, 1.0))

    if math.hypot(tx, ty) > 1.0:  # only steer heading when actually moving
        desired_yaw = math.atan2(ty, tx)
        dyaw = _wrap(desired_yaw - agent_yaw)
        ayaw = float(np.clip(dyaw / (scale * 0.1), -1.0, 1.0))
    else:
        ayaw = 0.0

    return np.array([ax, ay, ayaw], dtype=np.float32)


def collect(out_path: str, n_demos: int, overwrite: bool) -> None:
    import pygame as pg

    cfg = Config.default()
    scale = cfg.env.action_scale

    if overwrite and os.path.exists(out_path):
        shutil.rmtree(out_path)
        print(f"[collect] removed existing store {out_path}")

    already = num_episodes(out_path) if os.path.exists(out_path) else 0
    if already:
        print(f"[collect] {out_path} already holds {already} demos — appending.")

    env = TrajectoryExploreEnv(cfg.env, render_mode="human")
    print(
        "[collect] move the mouse to drag the agent to the goal.\n"
        "          R = restart episode | N = save now | ESC = quit\n"
        f"[collect] target: {n_demos} new demos."
    )

    saved = 0
    quit_all = False
    while saved < n_demos and not quit_all:
        obs, _ = env.reset()
        states = [obs[:3].copy()]
        actions: list[np.ndarray] = []
        had_collision = False
        info = {"is_success": False, "path_len": 0.0, "collision": False}
        outcome = None  # "save" | "discard" | "done"

        while outcome is None:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    outcome, quit_all = "discard", True
                elif ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_ESCAPE:
                        outcome, quit_all = "discard", True
                    elif ev.key == pg.K_r:
                        outcome = "discard"
                    elif ev.key == pg.K_n:
                        outcome = "save"
            if outcome is not None:
                break

            agent_xy = (obs[0], obs[1])
            action = _mouse_action(agent_xy, obs[2], pg.mouse.get_pos(), scale)
            obs, _, terminated, truncated, info = env.step(action)

            states.append(obs[:3].copy())
            actions.append(action)
            had_collision = had_collision or bool(info["collision"])

            if terminated or truncated:
                outcome = "save"

        if outcome == "discard":
            print(f"[collect] episode discarded ({len(actions)} steps).")
            continue

        n = len(actions)
        if n < 2:
            print("[collect] episode too short — skipped.")
            continue

        states_arr = np.asarray(states[:n], dtype=np.float32)
        actions_arr = np.asarray(actions, dtype=np.float32)
        meta = {
            "is_success": bool(info.get("is_success", False)),
            "path_len": float(info.get("path_len", 0.0)),
            "n_steps": n,
            "had_collision": had_collision,
        }
        save_episode(out_path, states_arr, actions_arr, meta)
        saved += 1
        print(
            f"[collect] saved demo {saved}/{n_demos} "
            f"(success={meta['is_success']}, path_len={meta['path_len']:.0f}, "
            f"steps={n}, collision={had_collision})"
        )

    env.close()
    total = num_episodes(out_path) if os.path.exists(out_path) else 0
    print(f"[collect] done — {saved} new demo(s) this run, {total} total in {out_path}.")


def main() -> None:
    p = argparse.ArgumentParser(description="Collect human teleop demos (PushT-style mouse follow).")
    p.add_argument("--demos", type=int, default=8, help="number of demos to collect this run")
    p.add_argument("--out", type=str, default=Config.default().buffer.demo_path,
                   help="output Zarr store path")
    p.add_argument("--overwrite", action="store_true",
                   help="delete the existing store before collecting")
    args = p.parse_args()

    if not _has_display():
        print(
            "No graphical display detected (DISPLAY/WAYLAND_DISPLAY not set).\n"
            "Run this script on a machine with a display or via X forwarding."
        )
        sys.exit(1)

    collect(args.out, args.demos, args.overwrite)


if __name__ == "__main__":
    main()
