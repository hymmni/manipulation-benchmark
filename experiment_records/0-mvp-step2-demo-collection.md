# Step 2: demo-collection

**Date**: 2026-06-04

## Summary
Implemented human demo collection infrastructure for the 0-mvp phase.

## Deliverables

### `data/spline.py`
- `interpolate_waypoints(waypoints, n_points, kind)`: (K,2) → (n_points,3) float32
- Uses `scipy.interpolate.CubicSpline` (bc_type="not-a-knot") for C2-smooth paths
- yaw = `atan2(dy, dx)` of spline tangent, then `np.unwrap` to remove ±π discontinuities
- Falls back to linear interp for K==2

### `data/zarr_io.py`
- Schema (diffusion-policy convention):
  - `/states (total_T, 3) float32`
  - `/actions (total_T, 3) float32`
  - `/episode_ends (n_ep,) int64` — cumulative end indices
  - `/meta/<key>` — scalar arrays per meta key
- Append-safe: creates store on first call, appends on subsequent calls
- Default paths: `BufferConfig.demo_path = data_store/demos.zarr`, `self_path = data_store/self_collected.zarr`

### `data/demo_collector.py`
- `DemoCollector(cfg, store_path)`: pygame GUI for mouse-click waypoint collection
- `build_trajectory(waypoints)`: pure logic (testable headlessly)
- `run()`: interactive loop with live spline preview; guarded against headless execution

### `collect_demos.py`
- Thin CLI entry point: `Config.default()` → `DemoCollector.run()`
- Exits gracefully with message if no display

### `.gitignore`
- Added `data_store/` and `*.zarr` entries (ADR-002)

## Test Results
- 8 new tests in `tests/test_spline.py` — all passed
- 36 total tests passed (no regressions)

## AC Verification
```
(150, 3) float32 0.029880085960030556
```
Shape correct, dtype float32, max yaw jump 0.030 rad << π.
