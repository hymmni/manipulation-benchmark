import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Tuple

Point = Tuple[float, float]
_NS = "http://www.w3.org/2000/svg"
_TAG = "{" + _NS + "}"


# ── geometry helpers ──────────────────────────────────────────────────────────

def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_seg(p: Point, q: Point, r: Point) -> bool:
    return (
        min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
        and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    )


def _seg_seg(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1-p2 properly intersects segment p3-p4."""
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    if d1 == 0 and _on_seg(p3, p1, p4):
        return True
    if d2 == 0 and _on_seg(p3, p2, p4):
        return True
    if d3 == 0 and _on_seg(p1, p3, p2):
        return True
    if d4 == 0 and _on_seg(p1, p4, p2):
        return True
    return False


def _dist_pt_seg(px: float, py: float, p0: Point, p1: Point) -> float:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    len2 = dx * dx + dy * dy
    if len2 == 0:
        return math.hypot(px - p0[0], py - p0[1])
    t = max(0.0, min(1.0, ((px - p0[0]) * dx + (py - p0[1]) * dy) / len2))
    return math.hypot(px - (p0[0] + t * dx), py - (p0[1] + t * dy))


# ── SVG transform parsing ─────────────────────────────────────────────────────

def _parse_transform(s: str) -> List[Tuple[str, List[float]]]:
    ops = []
    for m in re.finditer(r'(translate|rotate|scale)\(([^)]*)\)', s):
        nums = [float(v) for v in re.split(r'[\s,]+', m.group(2).strip()) if v]
        ops.append((m.group(1), nums))
    return ops


def _apply_transform(x: float, y: float, ops: List[Tuple[str, List[float]]]) -> Point:
    for kind, args in ops:
        if kind == "translate":
            x += args[0]
            y += args[1] if len(args) > 1 else 0.0
        elif kind == "rotate":
            a = math.radians(args[0])
            cx = args[1] if len(args) > 1 else 0.0
            cy = args[2] if len(args) > 2 else 0.0
            ca, sa = math.cos(a), math.sin(a)
            dx, dy = x - cx, y - cy
            x = dx * ca - dy * sa + cx
            y = dx * sa + dy * ca + cy
        elif kind == "scale":
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            x *= sx
            y *= sy
    return x, y


def _txpts(pts: List[Point], ops) -> List[Point]:
    return [_apply_transform(x, y, ops) for x, y in pts]


# ── Obstacle types ────────────────────────────────────────────────────────────

class Obstacle:
    def contains(self, x: float, y: float) -> bool:
        raise NotImplementedError

    def intersects_segment(self, p0: Point, p1: Point) -> bool:
        raise NotImplementedError


@dataclass
class Polygon(Obstacle):
    vertices: List[Point]

    def contains(self, x: float, y: float) -> bool:
        inside = False
        vx, vy = self.vertices[-1]
        for wx, wy in self.vertices:
            if (vy > y) != (wy > y) and x < (wx - vx) * (y - vy) / (wy - vy) + vx:
                inside = not inside
            vx, vy = wx, wy
        return inside

    def intersects_segment(self, p0: Point, p1: Point) -> bool:
        if self.contains(*p0) or self.contains(*p1):
            return True
        n = len(self.vertices)
        for i in range(n):
            if _seg_seg(p0, p1, self.vertices[i], self.vertices[(i + 1) % n]):
                return True
        return False


@dataclass
class Circle(Obstacle):
    cx: float
    cy: float
    r: float

    def contains(self, x: float, y: float) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2

    def intersects_segment(self, p0: Point, p1: Point) -> bool:
        return _dist_pt_seg(self.cx, self.cy, p0, p1) <= self.r


@dataclass
class Ellipse(Obstacle):
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float = 0.0  # rotation in radians

    def contains(self, x: float, y: float) -> bool:
        ca, sa = math.cos(-self.angle), math.sin(-self.angle)
        dx, dy = x - self.cx, y - self.cy
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        return (lx / self.rx) ** 2 + (ly / self.ry) ** 2 <= 1.0

    def intersects_segment(self, p0: Point, p1: Point) -> bool:
        if self.contains(*p0) or self.contains(*p1):
            return True
        for t in (i / 20 for i in range(1, 20)):
            mx = p0[0] + t * (p1[0] - p0[0])
            my = p0[1] + t * (p1[1] - p0[1])
            if self.contains(mx, my):
                return True
        return False


# ── MapSpec ───────────────────────────────────────────────────────────────────

@dataclass
class MapSpec:
    width: int
    height: int
    obstacles: List[Obstacle]
    agent_start: Tuple[float, float, float]   # x, y, yaw
    goal_center: Tuple[float, float]
    goal_polyline: List[Point]                 # 'ㄷ' world coords


# ── Fallback map (hard-coded from references/image.svg spec) ──────────────────

def _make_fallback() -> MapSpec:
    def _rot(pts, deg, cx, cy):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        return [(((x - cx) * ca - (y - cy) * sa + cx),
                 ((x - cx) * sa + (y - cy) * ca + cy))
                for x, y in pts]

    def _rect_pts(x, y, w, h):
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    obs: List[Obstacle] = [
        # row 1
        Polygon([(260, 90), (220, 160), (300, 160)]),
        Polygon(_rect_pts(380, 95, 75, 60)),
        Polygon(_rot([(650, 60), (600, 130), (700, 130)], 180, 650, 95)),
        Circle(730, 160, 28),
        # row 2
        Polygon([(270, 230), (220, 270), (270, 310)]),
        Polygon(_rect_pts(390, 225, 100, 65)),
        Polygon(_rot([(530, 220), (480, 290), (530, 290)], -30, 505, 255)),
        Circle(640, 245, 32),
        # row 3
        Polygon(_rect_pts(195, 420, 50, 50)),
        Polygon(_rect_pts(180, 500, 130, 50)),
        Circle(300, 415, 24),
        Polygon([(450, 380), (410, 450), (490, 450)]),
        Ellipse(460, 510, 55, 28),
        Polygon(_rot(_rect_pts(520, 425, 110, 55), 90, 575, 452.5)),
        Polygon(_rect_pts(495, 525, 115, 50)),
        Polygon([(740, 450), (690, 530), (790, 530)]),
    ]
    # goal polyline: path M40,-60 ... translated by (800,300)
    local_pts = [(40, -60), (-30, -60), (-30, -30), (10, -30),
                 (10, 30), (-30, 30), (-30, 60), (40, 60)]
    goal_poly = [(x + 800, y + 300) for x, y in local_pts]
    return MapSpec(
        width=900, height=600,
        obstacles=obs,
        agent_start=(130.0, 300.0, 0.0),
        goal_center=(800.0, 300.0),
        goal_polyline=goal_poly,
    )


FALLBACK_MAP: MapSpec = _make_fallback()


# ── SVG parser ────────────────────────────────────────────────────────────────

def _parse_points(s: str) -> List[Point]:
    nums = [float(v) for v in re.split(r'[\s,]+', s.strip()) if v]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _parse_path_polyline(d: str) -> List[Point]:
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', d)
    return [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums) - 1, 2)]


def _rect_to_polygon(x, y, w, h, ops) -> Polygon:
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return Polygon(_txpts(corners, ops))


def load_map(svg_path: str) -> MapSpec:
    try:
        return _load_svg(svg_path)
    except Exception:
        return FALLBACK_MAP


def _load_svg(svg_path: str) -> MapSpec:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # viewBox → width/height
    vb = root.get("viewBox", "0 0 900 600").split()
    width, height = int(float(vb[2])), int(float(vb[3]))

    obstacles: List[Obstacle] = []
    agent_start: Tuple[float, float, float] = (130.0, 300.0, 0.0)
    goal_center: Tuple[float, float] = (800.0, 300.0)
    goal_polyline: List[Point] = []

    def _is_obstacle_tag(tag: str) -> bool:
        local = tag.replace(_TAG, "")
        return local in ("polygon", "rect", "circle", "ellipse")

    # IDs of groups to skip when collecting obstacles
    skip_ids = {"agent_robot", "target_goal"}

    # IDs encountered in groups
    group_ids_seen = set()
    for g in root.findall(f"{_TAG}g"):
        gid = g.get("id", "")
        group_ids_seen.add(gid)
        g_ops = _parse_transform(g.get("transform", ""))

        if gid == "agent_robot":
            # translate gives start position; yaw = 0 (arrow points +x)
            for op_kind, args in g_ops:
                if op_kind == "translate":
                    agent_start = (args[0], args[1] if len(args) > 1 else 0.0, 0.0)
            continue

        if gid == "target_goal":
            for op_kind, args in g_ops:
                if op_kind == "translate":
                    goal_center = (args[0], args[1] if len(args) > 1 else 0.0)
            path_el = g.find(f"{_TAG}path")
            if path_el is not None:
                local_pts = _parse_path_polyline(path_el.get("d", ""))
                goal_polyline = [_apply_transform(x, y, g_ops) for x, y in local_pts]
            continue

    # Collect direct obstacle children (not inside agent/goal groups)
    for child in root:
        tag_local = child.tag.replace(_TAG, "")
        if not _is_obstacle_tag(child.tag):
            continue
        ops = _parse_transform(child.get("transform", ""))

        if tag_local == "polygon":
            verts = _txpts(_parse_points(child.get("points", "")), ops)
            obstacles.append(Polygon(verts))

        elif tag_local == "rect":
            x = float(child.get("x", 0))
            y = float(child.get("y", 0))
            w = float(child.get("width", 0))
            h = float(child.get("height", 0))
            obstacles.append(_rect_to_polygon(x, y, w, h, ops))

        elif tag_local == "circle":
            cx = float(child.get("cx", 0))
            cy = float(child.get("cy", 0))
            r = float(child.get("r", 0))
            # circles have no transform other than translate usually; apply center transform
            ncx, ncy = _apply_transform(cx, cy, ops)
            obstacles.append(Circle(ncx, ncy, r))

        elif tag_local == "ellipse":
            cx = float(child.get("cx", 0))
            cy = float(child.get("cy", 0))
            rx = float(child.get("rx", 0))
            ry = float(child.get("ry", 0))
            ncx, ncy = _apply_transform(cx, cy, ops)
            obstacles.append(Ellipse(ncx, ncy, rx, ry))

    if not obstacles:
        raise ValueError("No obstacles found in SVG")

    return MapSpec(
        width=width,
        height=height,
        obstacles=obstacles,
        agent_start=agent_start,
        goal_center=goal_center,
        goal_polyline=goal_polyline,
    )
