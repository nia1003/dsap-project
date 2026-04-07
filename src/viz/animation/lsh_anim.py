"""
Manim animation: LSH — random projection hashing and bucket retrieval.

Shows:
  1. HashScene  — random hyperplanes slice the space; points fall into buckets
  2. QueryScene — query point is hashed, its bucket lights up, candidates retrieved

Usage:
    manim -pql src/viz/animation/lsh_anim.py HashScene
    manim -pql src/viz/animation/lsh_anim.py QueryScene
    manim -pqh src/viz/animation/lsh_anim.py QueryScene
"""

from __future__ import annotations
import numpy as np
from manim import *

RNG = np.random.default_rng(42)
N_POINTS = 40
POINTS_2D = RNG.uniform(-3.2, 3.2, (N_POINTS, 2))

BG_COLOR      = "#0a0a1a"
POINT_COLOR   = "#4C9BE8"
PLANE_COLORS  = ["#E8844C", "#55A868", "#C770CF"]
BUCKET_COLORS = ["#1a3a5c", "#1a4a2a", "#3a1a4a", "#4a3a1a",
                 "#2a1a4a", "#1a4a3a", "#4a1a2a", "#3a4a1a"]
QUERY_COLOR   = WHITE
HIT_COLOR     = "#FFD700"


def _project(points, normal):
    """Return sign of projection of points onto normal. Shape (N,)."""
    return (points @ normal >= 0).astype(int)


def _hash_key(points, planes):
    """Return integer bucket key per point. planes shape: (n_bits, 2)."""
    bits = np.stack([_project(points, p) for p in planes], axis=1)  # (N, n_bits)
    powers = 2 ** np.arange(bits.shape[1])
    return (bits * powers).sum(axis=1)


# ── Scene 1: Hashing ──────────────────────────────────────────────────────────

class HashScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("LSH: Random Projection Hashing", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        # Draw points
        dots = VGroup(*[
            Dot([p[0], p[1], 0], radius=0.07, color=POINT_COLOR)
            for p in POINTS_2D
        ])
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.04), run_time=1.2)
        self.wait(0.3)

        planes_2d = RNG.standard_normal((2, 2)).astype(float)
        planes_2d /= np.linalg.norm(planes_2d, axis=1, keepdims=True)

        for bit_idx, (normal, color) in enumerate(zip(planes_2d, PLANE_COLORS)):
            # Draw hyperplane (line through origin, perpendicular to normal)
            perp = np.array([-normal[1], normal[0]])
            scale = 5.0
            line = Line(
                (-perp * scale).tolist() + [0],
                ( perp * scale).tolist() + [0],
                color=color, stroke_width=2.5,
            )
            label = Text(f"plane {bit_idx+1}", font_size=18, color=color)
            label.next_to(line.get_end(), RIGHT, buff=0.1)

            self.play(Create(line), FadeIn(label), run_time=0.6)

            # Color points by which side they fall on
            sides = _project(POINTS_2D, normal)
            anims = []
            for i, (dot, side) in enumerate(zip(dots, sides)):
                c = color if side == 1 else ManimColor(color).interpolate(BLACK, 0.5)
                anims.append(dot.animate.set_color(c))
            self.play(*anims, run_time=0.5)
            self.wait(0.3)

        self.wait(1)

        # Show final bucket coloring
        keys = _hash_key(POINTS_2D, planes_2d)
        unique_keys = sorted(set(keys))
        key_to_color = {k: BUCKET_COLORS[i % len(BUCKET_COLORS)]
                        for i, k in enumerate(unique_keys)}

        bucket_anims = [
            dots[i].animate.set_color(key_to_color[keys[i]])
            for i in range(N_POINTS)
        ]
        self.play(*bucket_anims, run_time=0.8)

        bucket_label = Text("Points in same bucket share the same hash",
                            font_size=20, color="#aaaaaa")
        bucket_label.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(bucket_label))
        self.wait(1.5)


# ── Scene 2: Query ────────────────────────────────────────────────────────────

class QueryScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("LSH: Query — Only Check One Bucket", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        planes_2d = RNG.standard_normal((2, 2)).astype(float)
        planes_2d /= np.linalg.norm(planes_2d, axis=1, keepdims=True)

        keys = _hash_key(POINTS_2D, planes_2d)
        unique_keys = sorted(set(keys))
        key_to_color = {k: BUCKET_COLORS[i % len(BUCKET_COLORS)]
                        for i, k in enumerate(unique_keys)}

        # Draw points colored by bucket
        dots = {}
        for i, p in enumerate(POINTS_2D):
            dots[i] = Dot([p[0], p[1], 0], radius=0.07,
                          color=key_to_color[keys[i]])
        self.play(LaggedStartMap(FadeIn, VGroup(*dots.values()), lag_ratio=0.04),
                  run_time=1.0)
        self.wait(0.3)

        # Drop query
        query = np.array([0.3, 0.5])
        q_dot = Dot([query[0], query[1], 0], radius=0.14, color=QUERY_COLOR)
        q_label = Text("query", font_size=18, color=WHITE).next_to(q_dot, UP, buff=0.1)
        self.play(FadeIn(q_dot, scale=2.5), FadeIn(q_label))
        self.wait(0.3)

        # Hash the query
        q_key = int(_hash_key(query[None], planes_2d)[0])
        hash_text = Text(f"hash(query) = {q_key:0{2}b} (bucket {q_key})",
                         font_size=20, color=QUERY_COLOR)
        hash_text.to_edge(DOWN, buff=0.7)
        self.play(FadeIn(hash_text))
        self.wait(0.4)

        # Dim everything not in the same bucket
        dim_anims = []
        hit_anims = []
        for i, dot in dots.items():
            if keys[i] == q_key:
                hit_anims.append(dot.animate.set_color(HIT_COLOR).scale(1.5))
            else:
                dim_anims.append(dot.animate.set_opacity(0.1))

        self.play(*dim_anims, run_time=0.5)
        self.play(*hit_anims, run_time=0.5)

        n_candidates = sum(1 for k in keys if k == q_key)
        candidate_text = Text(
            f"Only {n_candidates}/{N_POINTS} candidates checked  →  sub-linear!",
            font_size=22, color=HIT_COLOR,
        )
        candidate_text.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(candidate_text))
        self.wait(2)
