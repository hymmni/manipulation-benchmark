import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.config import EnvConfig
from .svg_map import load_map, MapSpec, Polygon, Circle, Ellipse
from .obstacles import ObstacleMap

_OBS_DIM = 5  # x, y, yaw, goal_dx, goal_dy


class TrajectoryExploreEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.render_mode = render_mode

        self._map: MapSpec = load_map(self.cfg.svg_path)
        self._obs_map = ObstacleMap(self._map.obstacles)

        w, h = float(self.cfg.width), float(self.cfg.height)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -math.pi, -w, -h], dtype=np.float32),
            high=np.array([w, h, math.pi, w, h], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # runtime state
        self._x: float = 0.0
        self._y: float = 0.0
        self._yaw: float = 0.0
        self._steps: int = 0
        self._path_len: float = 0.0

        # pygame handles
        self._pg = None
        self._screen = None

    # ── gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._x = float(self.cfg.agent_start[0])
        self._y = float(self.cfg.agent_start[1])
        self._yaw = float(self.cfg.agent_start[2])
        self._steps = 0
        self._path_len = 0.0

        if self.render_mode == "human":
            self._render_frame(display=True)

        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        scale = self.cfg.action_scale
        dx = float(action[0]) * scale
        dy = float(action[1]) * scale
        dyaw = float(action[2]) * scale * 0.1  # radians: 10× smaller than pixels

        new_x = np.clip(self._x + dx, 0.0, float(self.cfg.width))
        new_y = np.clip(self._y + dy, 0.0, float(self.cfg.height))
        new_yaw = self._yaw + dyaw
        # normalise yaw to (-π, π]
        new_yaw = (new_yaw + math.pi) % (2 * math.pi) - math.pi

        p0 = (self._x, self._y)
        p1 = (float(new_x), float(new_y))
        collision = self._obs_map.segment_collision(p0, p1)

        if collision:
            reward = -1.0
            # cancel movement
            new_x, new_y = self._x, self._y
        else:
            dist = math.hypot(new_x - self._x, new_y - self._y)
            self._path_len += dist
            self._x = float(new_x)
            self._y = float(new_y)
            self._yaw = new_yaw
            reward = 0.0

        self._steps += 1

        gx, gy = self.cfg.goal_center
        success = math.hypot(self._x - gx, self._y - gy) <= self.cfg.goal_radius
        terminated = bool(success)
        truncated = self._steps >= self.cfg.max_steps

        if success:
            reward = 10.0

        info: Dict[str, Any] = {
            "is_success": success,
            "path_len": self._path_len,
            "collision": collision,
        }

        if self.render_mode == "human":
            self._render_frame(display=True)

        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(display=False)
        elif self.render_mode == "human":
            self._render_frame(display=True)
            return None
        return None

    def close(self) -> None:
        if self._screen is not None and self._pg is not None:
            self._pg.display.quit()
            self._screen = None
        self._pg = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        gx, gy = self.cfg.goal_center
        obs = np.array(
            [self._x, self._y, self._yaw, gx - self._x, gy - self._y],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _ensure_pygame(self) -> Any:
        if self._pg is None:
            import pygame as pg  # lazy import — safe for headless
            if not pg.get_init():
                pg.init()
            self._pg = pg
        return self._pg

    def _render_frame(self, *, display: bool) -> Optional[np.ndarray]:
        pg = self._ensure_pygame()

        w, h = self.cfg.width, self.cfg.height
        surf = pg.Surface((w, h))
        surf.fill((250, 250, 250))

        # draw obstacles
        for obs in self._map.obstacles:
            if isinstance(obs, Polygon):
                pts = [(int(round(x)), int(round(y))) for x, y in obs.vertices]
                if len(pts) >= 3:
                    pg.draw.polygon(surf, (229, 231, 235), pts)
                    pg.draw.polygon(surf, (55, 65, 81), pts, 2)
            elif isinstance(obs, Circle):
                c = (int(round(obs.cx)), int(round(obs.cy)))
                pg.draw.circle(surf, (229, 231, 235), c, max(1, int(obs.r)))
                pg.draw.circle(surf, (55, 65, 81), c, max(1, int(obs.r)), 2)
            elif isinstance(obs, Ellipse):
                rect = pg.Rect(
                    int(round(obs.cx - obs.rx)),
                    int(round(obs.cy - obs.ry)),
                    max(1, int(2 * obs.rx)),
                    max(1, int(2 * obs.ry)),
                )
                pg.draw.ellipse(surf, (229, 231, 235), rect)
                pg.draw.ellipse(surf, (55, 65, 81), rect, 2)

        # draw goal
        if self._map.goal_polyline:
            pts = [(int(round(x)), int(round(y))) for x, y in self._map.goal_polyline]
            if len(pts) >= 3:
                pg.draw.polygon(surf, (147, 197, 253), pts)
                pg.draw.polygon(surf, (29, 78, 216), pts, 3)

        # draw agent
        ax, ay = int(round(self._x)), int(round(self._y))
        pg.draw.circle(surf, (220, 38, 38), (ax, ay), 28)
        pg.draw.circle(surf, (31, 41, 55), (ax, ay), 28, 3)
        tip_x = ax + int(28 * math.cos(self._yaw))
        tip_y = ay + int(28 * math.sin(self._yaw))
        pg.draw.line(surf, (31, 41, 55), (ax, ay), (tip_x, tip_y), 3)

        if display:
            if self._screen is None:
                self._screen = pg.display.set_mode((w, h))
                pg.display.set_caption("TrajectoryExplore")
            self._screen.blit(surf, (0, 0))
            pg.display.flip()
            pg.time.Clock().tick(self.metadata["render_fps"])
            return None

        # rgb_array: (H, W, 3)
        arr = pg.surfarray.array3d(surf)  # (W, H, 3)
        return arr.transpose(1, 0, 2).astype(np.uint8)
