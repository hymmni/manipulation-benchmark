import os

import numpy as np

from config.config import EnvConfig
from .spline import interpolate_waypoints
from .zarr_io import save_episode


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class DemoCollector:
    def __init__(self, cfg: EnvConfig, store_path: str) -> None:
        self.cfg = cfg
        self.store_path = store_path

    def build_trajectory(self, waypoints: np.ndarray) -> np.ndarray:
        """Pure logic: (K,2) waypoints → (n_points, 3) trajectory. Testable headlessly."""
        return interpolate_waypoints(waypoints, n_points=200)

    def run(self) -> None:
        """Interactive pygame GUI for demo collection.

        Left-click to add waypoints, ENTER to commit trajectory and save,
        R to reset current waypoints, Q/Escape to quit.
        """
        if not _has_display():
            raise RuntimeError(
                "No display detected (DISPLAY/WAYLAND_DISPLAY not set). "
                "Run demo_collector on a machine with a graphical display."
            )

        import pygame as pg  # lazy import — safe in headless env at import time

        from env.trajectory_explore_env import TrajectoryExploreEnv

        env = TrajectoryExploreEnv(self.cfg, render_mode=None)
        env.reset()

        pg.init()
        screen = pg.display.set_mode((self.cfg.width, self.cfg.height))
        pg.display.set_caption("Demo Collector — click waypoints, ENTER to save, R to reset, Q to quit")
        clock = pg.time.Clock()

        waypoints: list[tuple[float, float]] = []
        traj: np.ndarray | None = None
        demo_count = 0

        def render_base() -> pg.Surface:
            frame = env._render_frame(display=False)
            surf = pg.surfarray.make_surface(frame.transpose(1, 0, 2))
            return surf

        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key in (pg.K_q, pg.K_ESCAPE):
                        running = False
                    elif event.key == pg.K_r:
                        waypoints.clear()
                        traj = None
                    elif event.key == pg.K_RETURN and len(waypoints) >= 2:
                        wp = np.array(waypoints, dtype=float)
                        traj = self.build_trajectory(wp)
                        states = traj
                        save_episode(self.store_path, states, meta={"is_demo": True, "n_steps": len(states)})
                        demo_count += 1
                        print(f"[DemoCollector] saved demo #{demo_count} ({len(states)} steps) → {self.store_path}")
                        waypoints.clear()
                        traj = None
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    waypoints.append((float(event.pos[0]), float(event.pos[1])))

            base = render_base()
            screen.blit(base, (0, 0))

            # draw waypoints
            for wp in waypoints:
                pg.draw.circle(screen, (0, 200, 0), (int(wp[0]), int(wp[1])), 6)
            if len(waypoints) >= 2:
                pg.draw.lines(screen, (0, 200, 0), False, [(int(x), int(y)) for x, y in waypoints], 2)

            # draw spline preview when ≥2 waypoints
            if len(waypoints) >= 2:
                try:
                    preview = self.build_trajectory(np.array(waypoints, dtype=float))
                    pts = [(int(p[0]), int(p[1])) for p in preview]
                    if len(pts) >= 2:
                        pg.draw.lines(screen, (255, 140, 0), False, pts, 2)
                except Exception:
                    pass

            font = pg.font.SysFont(None, 24)
            txt = font.render(
                f"Demos: {demo_count}  Waypoints: {len(waypoints)}  [ENTER] save  [R] reset  [Q] quit",
                True, (30, 30, 30),
            )
            screen.blit(txt, (10, 10))
            pg.display.flip()
            clock.tick(30)

        pg.quit()
        env.close()
