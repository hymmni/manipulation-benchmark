# Teleop demo collection — PushT-style mouse following

## Motivation
The "inefficient human demo" premise was filled by `main._make_synthetic_demos`, which
programmatically generates detour trajectories. To collect *real* human demos (and to
make demo inefficiency genuinely human-driven rather than scripted), we add a manual
teleoperation interface.

Decisions (confirmed with user):
- **Control scheme**: mouse-following only (PushT convention) — no keyboard.
- **Yaw**: auto-aligned to the direction of motion. The `yaw` channel stays in the
  state (3D state preserved) so CVAE / planner / IDM need no changes; only *how* the
  human sets yaw changes.

## What changed
- **Replaced** the root `collect_demos.py` (formerly a click-waypoint + spline GUI via
  `data/demo_collector.py`, which never recorded real actions — zero-filled) with a
  pygame teleop loop:
  - `action[:2] = clip((cursor - agent) / action_scale, -1, 1)` (move toward cursor).
  - `action[2]` = yaw increment toward `atan2(dy, dx)`, normalised by `scale*0.1`
    (env applies `dyaw = action[2] * scale * 0.1`); held at 0 when ~stationary.
  - Records `(states, actions, meta)` into the **same Zarr schema** as the training
    pipeline (`data.zarr_io.save_episode`), so no model/training code changes.
  - Controls: mouse drag, `R` restart episode, `N` save now, `ESC`/quit stop.
- **Removed**: `data/demo_collector.py` (click-waypoint GUI, superseded; no other code
  referenced it).
- **Unchanged**: env action space `(dx, dy, dyaw)`, all models, replay buffer,
  `data/spline.py` (still used by `main._make_synthetic_demos`).
- `_make_synthetic_demos` retained as the headless/server fallback.

## Why it lives at the repo root, not `scripts/`
`scripts/`, `docs/`, `.claude/`, `phases/` are git-ignored by harness design (local-only,
confirmed earlier: "harness는 로컬에만 두는게 설계 방향"). A demo-collection tool needs to
ship with the repo so it's available wherever the project is cloned, so it stays at the
tracked repo root — same convention as the file it replaced.

## Usage
Run on a machine **with a display** (local PC). Do NOT set `SDL_VIDEODRIVER=dummy`.
```bash
python collect_demos.py --demos 8                 # append to data_store/demos.zarr
python collect_demos.py --demos 8 --overwrite      # fresh store
```
Then transfer `data_store/demos.zarr` to the GPU server for pretraining.

## Verification
Headless unit check of the action mapping (`_mouse_action`, `_wrap`):
cursor-right → `[1,0,0]`; cursor-up → `[0,-1,-1]` (yaw→-π/2); diagonal → `[1,1,0.785]`;
cursor-on-agent → ~stationary. Imports OK in the `trajectory-explore` conda env.
The interactive mouse/event loop is display-only and exercised manually.
