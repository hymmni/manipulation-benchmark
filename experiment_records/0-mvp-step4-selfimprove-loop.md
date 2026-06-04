# Step 4: Self-Improve Loop — Experiment Record

**Date**: 2026-06-04
**Branch**: feat/0-mvp
**Phase**: 0-mvp (final step)

## Summary

Completed the self-growth online loop that closes the full MVP pipeline:
human demos → pretrain → rollout (exploration) → efficiency filter → self buffer → finetune → evaluate.

`python main.py --dry-run` completes 1 iteration on CPU in <5 seconds.
All 59 tests pass.

## Deliverables

| File | Role |
|---|---|
| `data/replay_buffer.py` | Sliding-window sampler on zarr store; episode-boundary-safe |
| `data/self_collected_buffer.py` | Subclass for high-quality shortcut episodes |
| `trainer/efficiency_filter.py` | `baseline_cost` + `is_efficient` (success & cost & no-collision) |
| `trainer/trainer.py` | `Trainer`: `build_models / pretrain / finetune / rollout / evaluate` |
| `main.py` | Self-improve loop entry point with `--dry-run` |
| `tests/test_loop.py` | 12 smoke tests covering buffer, filter, and dry-run integration |

## Loop Architecture

```
human demos (zarr)
      │
      ▼
  [1] pretrain (BC on demos)  ← CVAE + Planner + IDM
      │
      ▼  baseline_cost(demos)
      │
 ┌────▼─────────────────────────────────────────────────────────────────┐
 │  Self-Grow Iteration                                                  │
 │                                                                       │
 │  (a) Rollout ── Planner (CFG) plans waypoints                         │
 │                 IDM (exploration_noise) bridges → diverse actions      │
 │  (b) Filter  ── success & no-collision & cost < baseline*(1-margin)   │
 │  (c) Buffer  ── shortcuts → SelfCollectedBuffer                        │
 │  (d) Finetune── demo + self mixed replay (behavior cloning / BC)      │
 │  (e) Evaluate── success_rate, mean_cost logged                        │
 └───────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Why real-env rollout beats discriminator (vs. AdaptDiffuser)
AdaptDiffuser generates trajectories without an env → needs a discriminator to filter
OOD/infeasible trajectories. This project rolls out in the **actual 2-D env**, so physical
feasibility is guaranteed by the env itself. The MVP filter (success + efficiency + no-collision)
is sufficient without a learned discriminator; a discriminator hook is left in `is_efficient`
for future extension.

### Episode-boundary safety in ReplayBuffer
A window of size `max(horizon, action_horizon)` must lie entirely within one episode.
Cross-boundary windows would concatenate trajectories from different demos — physically
meaningless and harmful to training.

### DPPO upgrade path
`Trainer.finetune` uses behaviour cloning (diffusion noise prediction loss) on shortcut
episodes. A comment marks the DPPO (Diffusion Policy Policy Optimization, ICLR'25) upgrade:
treating the reverse diffusion chain as an MDP and applying policy gradient fine-tuning.

## Dry-run Result

```
[dry-run] temp store: /tmp/...
[loop] device=cpu
[dry-run] synthesised 3 demo episodes (7 steps each)
[loop] demo_buf windows=15, self_buf windows=0
[loop] models built
[loop] pretrain done: {'cvae': ..., 'planner': ..., 'idm': ...}
[loop] demo baseline cost = 690.08
[loop] iter=0 | rolled=2 | shortcuts=0 | success=0
[loop] iter=0 | success_rate=0 | mean_cost=nan | shortcuts_total=0 | finetune_loss_planner=...
[loop] done
```

No shortcuts found in the first iteration is expected — the untrained planner generates
random trajectories unlikely to reach the goal. After GPU pretraining on real demos,
the planner will generate goal-directed plans and the IDM exploration noise will start
discovering shorter paths.

## 0-mvp Phase Completion Summary

| Step | Name | Deliverables |
|---|---|---|
| 0 | project-setup | Config dataclasses, utils/device.py, package skeletons, environment.yml |
| 1 | env-2d | TrajectoryExploreEnv (gymnasium), SVG map, obstacle collision |
| 2 | demo-collection | Spline interpolation, Zarr I/O, DemoCollector CLI |
| 3 | models | CVAE, LatentTrajectoryDiffusion (CFG), InverseDynamicsDiffusion (exploration noise) |
| 4 | selfimprove-loop | ReplayBuffer, SelfCollectedBuffer, efficiency filter, Trainer, main loop |

**Total tests**: 59 passed.
