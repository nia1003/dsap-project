"""
Flat Search (Brute-Force)

Computes cosine similarity between the query and every embedding in the database.
Guarantees exact Top-K results — used as ground truth for recall evaluation.

Time complexity:
  - Build: O(1)
  - Query: O(n * d)  where n = database size, d = embedding dimension
"""

import numpy as np


class FlatSearch:
    def __init__(self):
        self._embeddings: np.ndarray | None = None  # (N, D)

    def build(self, embeddings: np.ndarray) -> None:
        """Store L2-normalised embeddings for cosine similarity search."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._embeddings = embeddings / np.where(norms == 0, 1, norms)

    def query(self, q: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Return indices and cosine similarities of the Top-K nearest neighbours.

        Args:
            q: (D,) query embedding (need not be normalised)
            k: number of neighbours to return

        Returns:
            indices:       (k,) int   — indices into the database
            similarities:  (k,) float — cosine similarity scores, descending
        """
        q_norm = q / (np.linalg.norm(q) or 1.0)
        sims = self._embeddings @ q_norm          # (N,)
        top_k = np.argpartition(sims, -k)[-k:]   # unordered Top-K
        top_k = top_k[np.argsort(sims[top_k])[::-1]]
        return top_k, sims[top_k]
