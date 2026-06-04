import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_waypoints(
    waypoints: np.ndarray,
    n_points: int = 200,
    kind: str = "cubic",
) -> np.ndarray:
    """Interpolate (K,2) waypoints into (n_points, 3) trajectory with smooth yaw.

    Returns float32 array of (x, y, yaw) where yaw is the tangent direction,
    unwrapped to avoid ±π discontinuities.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("waypoints must be shape (K, 2)")
    K = len(waypoints)
    if K < 2:
        raise ValueError("need at least 2 waypoints")

    t = np.zeros(K)
    for i in range(1, K):
        d = np.linalg.norm(waypoints[i] - waypoints[i - 1])
        t[i] = t[i - 1] + max(d, 1e-6)

    t_query = np.linspace(t[0], t[-1], n_points)

    if kind == "cubic" and K >= 3:
        cs = CubicSpline(t, waypoints, bc_type="not-a-knot")
        pts = cs(t_query)
        dpts = cs(t_query, 1)
    else:
        # linear fallback for K==2 or kind=="linear"
        cs_x = CubicSpline(t, waypoints[:, 0], bc_type="not-a-knot") if K >= 3 else None
        x = np.interp(t_query, t, waypoints[:, 0])
        y = np.interp(t_query, t, waypoints[:, 1])
        pts = np.stack([x, y], axis=1)
        dx = np.gradient(x, t_query)
        dy = np.gradient(y, t_query)
        dpts = np.stack([dx, dy], axis=1)

    yaw = np.arctan2(dpts[:, 1], dpts[:, 0])
    yaw = np.unwrap(yaw)

    traj = np.stack([pts[:, 0], pts[:, 1], yaw], axis=1).astype(np.float32)
    return traj
