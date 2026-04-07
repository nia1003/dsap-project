"""
LSH — Locality-Sensitive Hashing (Random Projection)

Uses multiple hash tables, each with L random hyperplanes.
Vectors on the same side of all L hyperplanes share the same bucket.
Multiple tables increase recall by reducing the chance of missing a neighbour.

Time complexity:
  - Build: O(n * n_tables * n_bits)
  - Query: O(n_tables * n_bits + |bucket| * d)  — sub-linear in n
"""

import numpy as np
from collections import defaultdict


class LSH:
    def __init__(self, n_bits: int = 4, n_tables: int = 16, seed: int = 42):
        """
        Args:
            n_bits:   number of hyperplanes per table (bucket granularity)
            n_tables: number of independent hash tables (recall vs speed)
            seed:     random seed for reproducibility
        """
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.seed = seed

        self._planes: list[np.ndarray] = []   # (n_tables, n_bits, D)
        self._tables: list[dict] = []         # n_tables hash tables
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------ build

    def build(self, embeddings: np.ndarray) -> None:
        """Generate random hyperplanes and hash all embeddings into tables."""
        self._embeddings = embeddings.astype(np.float32)
        n, d = embeddings.shape
        rng = np.random.default_rng(self.seed)

        self._planes = []
        self._tables = []

        for _ in range(self.n_tables):
            # Each plane: (n_bits, D) — random unit normals
            planes = rng.standard_normal((self.n_bits, d)).astype(np.float32)
            planes /= np.linalg.norm(planes, axis=1, keepdims=True)
            self._planes.append(planes)

            # Hash all embeddings: sign of projection → binary code → int key
            projections = embeddings @ planes.T          # (N, n_bits)
            codes = (projections >= 0).astype(np.uint64) # (N, n_bits)
            keys = self._codes_to_keys(codes)            # (N,)

            table: dict[int, list[int]] = defaultdict(list)
            for idx, key in enumerate(keys):
                table[key].append(idx)
            self._tables.append(table)

    # ------------------------------------------------------------------ query

    def query(self, q: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve candidates from all hash tables, then rank by cosine similarity.

        Returns:
            indices:      (k,) int
            similarities: (k,) float — cosine similarity, descending
        """
        q = q.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) or 1.0)

        candidates: set[int] = set()
        for t in range(self.n_tables):
            proj = self._planes[t] @ q                    # (n_bits,)
            code = (proj >= 0).astype(np.uint64)          # (n_bits,)
            key = int(self._codes_to_keys(code[None])[0])
            candidates.update(self._tables[t].get(key, []))

        if not candidates:
            # Fallback: return first k indices (edge case with very sparse buckets)
            candidates = set(range(min(k, len(self._embeddings))))

        cand = np.array(sorted(candidates), dtype=np.int64)
        sims = self._embeddings[cand] @ q_norm            # cosine similarity

        top_k_local = np.argpartition(sims, -min(k, len(sims)))[-min(k, len(sims)):]
        top_k_local = top_k_local[np.argsort(sims[top_k_local])[::-1]]

        indices = cand[top_k_local]
        return indices, sims[top_k_local]

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _codes_to_keys(codes: np.ndarray) -> np.ndarray:
        """Pack binary code array (N, n_bits) into integer keys (N,)."""
        n_bits = codes.shape[-1]
        powers = (1 << np.arange(n_bits, dtype=np.uint64))
        return (codes * powers).sum(axis=-1)
