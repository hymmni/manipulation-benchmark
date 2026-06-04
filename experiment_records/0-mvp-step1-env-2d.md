# Step 1: env-2d

**Date**: 2026-06-04

## Summary

Implemented the 2D continuous-space `gymnasium.Env` for the trajectory exploration benchmark.

## Deliverables

| File | Description |
|------|-------------|
| `env/svg_map.py` | SVG parser + obstacle dataclasses (`Polygon`, `Circle`, `Ellipse`) with `contains` / `intersects_segment` collision methods; `MapSpec` dataclass; hard-coded `FALLBACK_MAP`; `load_map()` entry point |
| `env/obstacles.py` | `ObstacleMap` wrapper providing batch `contains` and `segment_collision` queries |
| `env/trajectory_explore_env.py` | `TrajectoryExploreEnv(gymnasium.Env)` — obs `(5,) float32`, action `(3,) float32` in `[-1,1]`; headless-safe pygame render (lazy import, `rgb_array` always works) |
| `tests/test_env.py` | 13 new tests; gymnasium `check_env` passes |

## Interface Details

- **observation_space**: `Box(5,) float32` = `(x, y, yaw, goal_dx, goal_dy)`
- **action_space**: `Box(3,) float32` = `(dx, dy, dyaw)` ∈ `[-1, 1]`, scaled by `EnvConfig.action_scale=10.0`
- **info keys**: `is_success`, `path_len`, `collision`
- **render modes**: `"human"` (pygame window), `"rgb_array"` → `(600, 900, 3) uint8`

## Design Decisions

- **Coordinate system**: SVG y-down, origin top-left. No coordinate transform — SVG coords = world coords, simplifying parsing/debugging.
- **action_dim=3 alignment**: `(dx, dy, dyaw)` matches `ModelConfig.action_dim=3`.
- **SVG as single source of truth**: Parser reads `references/image.svg` directly; `FALLBACK_MAP` activates only on parse failure.
- **Headless safety**: `pygame` imported lazily inside `_ensure_pygame()`. `rgb_array` mode uses off-screen `pygame.Surface` — works with `SDL_VIDEODRIVER=dummy`.
- **Collision**: polygon ray-cast, circle distance-to-segment, ellipse approximate (normalized ellipse equation + 20-point sampling).
- **Rotated rects**: converted to `Polygon` at parse time by applying `rotate()` transform to all 4 corners.

## Test Results

```
28 passed, 1 warning in 0.59s
```
