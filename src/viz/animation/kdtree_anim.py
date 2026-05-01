"""
Manim animation: KD-Tree build and query traversal (2D example).

Renders two scenes:
  1. BuildScene  — partition lines appear one by one as the tree is built
  2. QueryScene  — query point drops in, traversal path glows, pruned regions fade

Usage:
    manim -pql src/viz/animation/kdtree_anim.py BuildScene
    manim -pql src/viz/animation/kdtree_anim.py QueryScene
    manim -pqh src/viz/animation/kdtree_anim.py QueryScene   # high quality
"""

from __future__ import annotations
import numpy as np

try:
    from manim import *
except ImportError as exc:
    raise ImportError(
        "manim is required to render animations. "
        "Install the system dependency first (apt-get install libpangocairo-1.0-0), "
        "then: pip install manim"
    ) from exc

RNG = np.random.default_rng(7)
N_POINTS = 30
POINTS_2D = RNG.uniform(-3.5, 3.5, (N_POINTS, 2))

AXIS_COLOR   = "#888888"
POINT_COLOR  = "#4C9BE8"
SPLIT_X_COLOR = "#E8844C"
SPLIT_Y_COLOR = "#55A868"
QUERY_COLOR  = WHITE
NEIGHBOR_COLOR = "#FFD700"
PRUNE_COLOR  = "#333355"


# ── KD-Tree (2D, for animation) ───────────────────────────────────────────────

class _Node2D:
    def __init__(self, idx, axis, val, bbox, left=None, right=None):
        self.idx   = idx    # index into POINTS_2D
        self.axis  = axis   # 0=x, 1=y
        self.val   = val    # split value
        self.bbox  = bbox   # (xmin, xmax, ymin, ymax)
        self.left  = left
        self.right = right


def _build_2d(indices, bbox, depth=0):
    if len(indices) == 0:
        return None
    axis = depth % 2
    order = indices[np.argsort(POINTS_2D[indices, axis])]
    mid = len(order) // 2
    node_idx = order[mid]
    val = float(POINTS_2D[node_idx, axis])

    xmin, xmax, ymin, ymax = bbox
    if axis == 0:
        left_bbox  = (xmin, val,  ymin, ymax)
        right_bbox = (val,  xmax, ymin, ymax)
    else:
        left_bbox  = (xmin, xmax, ymin, val)
        right_bbox = (xmin, xmax, val,  ymax)

    return _Node2D(
        idx=node_idx, axis=axis, val=val, bbox=bbox,
        left=_build_2d(order[:mid],    left_bbox,  depth + 1),
        right=_build_2d(order[mid+1:], right_bbox, depth + 1),
    )


ROOT = _build_2d(np.arange(N_POINTS), (-3.8, 3.8, -3.8, 3.8))


def _collect_splits(node, splits=None):
    """DFS to collect (axis, val, bbox) in build order."""
    if node is None:
        return splits
    if splits is None:
        splits = []
    splits.append((node.axis, node.val, node.bbox))
    _collect_splits(node.left,  splits)
    _collect_splits(node.right, splits)
    return splits


def _search_path(node, query, k=1, path=None, pruned=None):
    """Return (visited node indices, pruned bboxes)."""
    if node is None:
        return path, pruned
    if path is None:
        path, pruned = [], []

    path.append(node.idx)
    diff = query[node.axis] - node.val
    near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)

    _search_path(near, query, k, path, pruned)

    # Simple heuristic: prune far if |diff| > some threshold
    if abs(diff) > 1.5:
        if far is not None:
            pruned.append(far.bbox)
    else:
        _search_path(far, query, k, path, pruned)

    return path, pruned


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dot(pos, color=POINT_COLOR, radius=0.07):
    return Dot(point=[pos[0], pos[1], 0], radius=radius, color=color)


def _split_line(axis, val, bbox):
    xmin, xmax, ymin, ymax = bbox
    if axis == 0:  # vertical
        return Line([val, ymin, 0], [val, ymax, 0],
                    color=SPLIT_X_COLOR, stroke_width=2)
    else:          # horizontal
        return Line([xmin, val, 0], [xmax, val, 0],
                    color=SPLIT_Y_COLOR, stroke_width=2)


def _bbox_rect(bbox, color=PRUNE_COLOR, opacity=0.35):
    xmin, xmax, ymin, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    rect = Rectangle(width=w, height=h, color=color, fill_color=color,
                     fill_opacity=opacity, stroke_width=0)
    rect.move_to([(xmin + xmax) / 2, (ymin + ymax) / 2, 0])
    return rect


# ── Scene 1: Build ────────────────────────────────────────────────────────────

class BuildScene(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a1a"

        title = Text("KD-Tree: Building the Index", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        legend = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=SPLIT_X_COLOR),
            Text(" split on X", font_size=18, color=SPLIT_X_COLOR),
            Line(ORIGIN, RIGHT * 0.5, color=SPLIT_Y_COLOR),
            Text(" split on Y", font_size=18, color=SPLIT_Y_COLOR),
        ).arrange(RIGHT, buff=0.2).to_edge(DOWN, buff=0.3)
        self.add(legend)

        # Draw all points
        dots = VGroup(*[_dot(p) for p in POINTS_2D])
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.05), run_time=1.5)
        self.wait(0.3)

        # Draw splits one by one
        splits = _collect_splits(ROOT)
        for axis, val, bbox in splits[:12]:   # show first 12 levels
            line = _split_line(axis, val, bbox)
            self.play(Create(line), run_time=0.4)

        self.wait(1)


# ── Scene 2: Query ────────────────────────────────────────────────────────────

class QueryScene(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a1a"

        title = Text("KD-Tree: Query & Pruning", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        # Draw splits
        splits = _collect_splits(ROOT)
        lines = VGroup(*[_split_line(a, v, b) for a, v, b in splits[:12]])
        self.play(LaggedStartMap(Create, lines, lag_ratio=0.03), run_time=1.0)

        # Draw points (dim)
        dots = {i: _dot(POINTS_2D[i], color="#2a4a6a") for i in range(N_POINTS)}
        self.play(LaggedStartMap(FadeIn, VGroup(*dots.values()), lag_ratio=0.02),
                  run_time=0.8)

        # Drop query point
        query = np.array([0.5, 0.8])
        q_dot = _dot(query, color=QUERY_COLOR, radius=0.12)
        q_label = Text("query", font_size=18, color=WHITE).next_to(q_dot, UP, buff=0.1)
        self.play(FadeIn(q_dot, scale=2), FadeIn(q_label))
        self.wait(0.3)

        # Traversal
        path_indices, pruned_bboxes = _search_path(ROOT, query)

        visited_label = Text("traversing...", font_size=18, color=SPLIT_X_COLOR)
        visited_label.to_edge(LEFT, buff=0.5).shift(DOWN * 0.5)
        self.play(FadeIn(visited_label))

        for pt_idx in path_indices:
            dot = dots[pt_idx]
            self.play(dot.animate.set_color(SPLIT_X_COLOR).scale(1.5), run_time=0.2)
            self.play(dot.animate.scale(1 / 1.5), run_time=0.1)

        # Pruned regions
        self.play(FadeOut(visited_label))
        prune_label = Text("pruned ✗", font_size=18, color="#aa4444")
        prune_label.to_edge(LEFT, buff=0.5).shift(DOWN * 0.5)
        self.play(FadeIn(prune_label))

        for bbox in pruned_bboxes:
            rect = _bbox_rect(bbox)
            self.play(FadeIn(rect), run_time=0.3)

        self.play(FadeOut(prune_label))

        # Highlight nearest neighbour
        dists = np.linalg.norm(POINTS_2D - query, axis=1)
        nn_idx = int(np.argmin(dists))
        nn_dot = dots[nn_idx]
        self.play(nn_dot.animate.set_color(NEIGHBOR_COLOR).scale(1.8), run_time=0.4)
        nn_label = Text("nearest!", font_size=18, color=NEIGHBOR_COLOR)
        nn_label.next_to(nn_dot, RIGHT, buff=0.15)
        self.play(FadeIn(nn_label))

        self.wait(1.5)
