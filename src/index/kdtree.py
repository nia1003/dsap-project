"""
KD-Tree (from scratch)

Recursively partitions embedding space by splitting on the dimension
with highest variance at each node. Query uses branch-and-bound pruning
to skip subtrees that cannot contain a closer neighbour.

Embeddings are L2-normalised at build time so that L2 distance is
monotone with cosine distance — making results comparable to FlatSearch.

Time complexity:
  - Build: O(n log n)
  - Query: O(log n) average, O(n) worst case (high dimensions)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class _Node:
    split_dim: int        # dimension used for splitting; -1 for leaf nodes
    split_val: float      # value at split
    idx: int = -1         # index of the median point (internal nodes only)
    left: "_Node | None" = field(default=None, repr=False)
    right: "_Node | None" = field(default=None, repr=False)
    leaf_indices: "np.ndarray | None" = field(default=None, repr=False)


class KDTree:
    def __init__(self, leaf_size: int = 10):
        """
        Args:
            leaf_size: stop splitting when a node has <= leaf_size points.
                       Larger values trade tree depth for simpler nodes.
        """
        self.leaf_size = leaf_size
        self._root: _Node | None = None
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------ build

    def build(self, embeddings: np.ndarray) -> None:
        """Build the KD-Tree from an (N, D) embedding array.

        Normalises embeddings so L2 distance equals cosine distance ordering.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._embeddings = (embeddings / np.where(norms == 0, 1, norms)).astype(np.float32)
        indices = np.arange(len(self._embeddings))
        self._root = self._build(indices)

    def _build(self, indices: np.ndarray) -> _Node | None:
        if len(indices) == 0:
            return None

        # Leaf node: store all remaining indices directly
        if len(indices) <= self.leaf_size:
            return _Node(split_dim=-1, split_val=0.0, leaf_indices=indices)

        data = self._embeddings[indices]

        # Choose split dimension: highest variance
        split_dim = int(np.argmax(np.var(data, axis=0)))
        order = np.argsort(data[:, split_dim])
        sorted_indices = indices[order]
        mid = len(sorted_indices) // 2

        node = _Node(
            split_dim=split_dim,
            split_val=float(self._embeddings[sorted_indices[mid], split_dim]),
            idx=sorted_indices[mid],
        )
        node.left = self._build(sorted_indices[:mid])
        node.right = self._build(sorted_indices[mid + 1:])
        return node

    # ------------------------------------------------------------------ query

    def query(self, q: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Return Top-K nearest neighbours using L2 distance + branch-and-bound.

        Returns:
            indices:   (k,) int
            distances: (k,) float — L2 distances, ascending
        """
        q = q.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) or 1.0)

        # Max-heap stored as list of (-dist, idx)
        heap: list[tuple[float, int]] = []
        self._search(self._root, q_norm, k, heap)

        heap.sort(key=lambda x: x[0], reverse=True)  # sort by -dist descending = closest first
        indices = np.array([i for _, i in heap], dtype=np.int64)
        distances = np.array([-d for d, _ in heap], dtype=np.float32)
        return indices, distances

    def _search(self, node: _Node | None, q: np.ndarray, k: int,
                heap: list) -> None:
        if node is None:
            return

        # Leaf node: evaluate all stored points
        if node.leaf_indices is not None:
            for idx in node.leaf_indices:
                dist = float(np.linalg.norm(self._embeddings[idx] - q))
                _heap_push(heap, (-dist, idx), k)
            return

        dist = float(np.linalg.norm(self._embeddings[node.idx] - q))
        _heap_push(heap, (-dist, node.idx), k)

        diff = q[node.split_dim] - node.split_val
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)

        self._search(near, q, k, heap)

        # Pruning: only explore far branch if the splitting hyperplane is
        # closer than the current k-th best distance.
        worst = -heap[0][0] if heap else float("inf")
        if abs(diff) < worst or len(heap) < k:
            self._search(far, q, k, heap)


# ------------------------------------------------------------------ heap utils

def _heap_push(heap: list, item: tuple, max_size: int) -> None:
    """Maintain a max-heap of size max_size (largest -dist = worst neighbour at top)."""
    import heapq
    heapq.heappush(heap, item)
    if len(heap) > max_size:
        heapq.heappop(heap)
