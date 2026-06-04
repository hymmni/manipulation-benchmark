from typing import List, Tuple

from .svg_map import Obstacle

Point = Tuple[float, float]


class ObstacleMap:
    """Batch collision queries over a list of obstacles."""

    def __init__(self, obstacles: List[Obstacle]) -> None:
        self.obstacles = obstacles

    def contains(self, x: float, y: float) -> bool:
        return any(obs.contains(x, y) for obs in self.obstacles)

    def segment_collision(self, p0: Point, p1: Point) -> bool:
        return any(obs.intersects_segment(p0, p1) for obs in self.obstacles)
