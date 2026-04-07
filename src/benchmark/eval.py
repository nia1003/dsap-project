"""
Benchmark framework

Measures two metrics for any index:
  - recall@k : fraction of true Top-K neighbours found
  - latency  : average query time in milliseconds

Usage:
    results = benchmark(index, embeddings, k=10, n_queries=100)
"""

import time
import numpy as np
from typing import Protocol


class Index(Protocol):
    def build(self, embeddings: np.ndarray) -> None: ...
    def query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


def compute_ground_truth(embeddings: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """
    Use exact cosine similarity to compute true Top-K neighbours for each query.

    Returns:
        gt: (n_queries, k) int — ground truth indices
    """
    from src.index.flat import FlatSearch
    flat = FlatSearch()
    flat.build(embeddings)
    gt = np.array([flat.query(q, k)[0] for q in queries])
    return gt


def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Fraction of ground truth neighbours found in predicted results.

    Args:
        predicted:    (k,)  int
        ground_truth: (k,)  int
    """
    return len(set(predicted) & set(ground_truth)) / len(ground_truth)


def benchmark(
    index: Index,
    embeddings: np.ndarray,
    k: int = 10,
    n_queries: int = 100,
    seed: int = 0,
) -> dict:
    """
    Build the index and evaluate recall@k and query latency.

    Args:
        index:      any object implementing build() and query()
        embeddings: (N, D) float32 database
        k:          number of neighbours
        n_queries:  number of random queries to sample

    Returns:
        dict with keys: recall_at_k, latency_ms_mean, latency_ms_std, n_queries, k
    """
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(embeddings), size=n_queries, replace=False)
    queries = embeddings[query_idx]

    # Ground truth from exact search
    gt = compute_ground_truth(embeddings, queries, k)

    # Build index
    index.build(embeddings)

    # Query and time
    recalls, latencies = [], []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        pred, _ = index.query(q, k)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
        recalls.append(recall_at_k(pred, gt[i]))

    return {
        "recall_at_k": float(np.mean(recalls)),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies)),
        "n_queries": n_queries,
        "k": k,
    }


def compare_all(
    indexes: dict,
    embeddings: np.ndarray,
    k: int = 10,
    n_queries: int = 100,
) -> dict[str, dict]:
    """
    Run benchmark on multiple indexes and return a results dict.

    Args:
        indexes: {"Flat": FlatSearch(), "KDTree": KDTree(), ...}

    Returns:
        {"Flat": {...}, "KDTree": {...}, ...}
    """
    results = {}
    for name, idx in indexes.items():
        print(f"Benchmarking {name}...")
        results[name] = benchmark(idx, embeddings, k=k, n_queries=n_queries)
        r = results[name]
        print(f"  recall@{k}: {r['recall_at_k']:.3f}  "
              f"latency: {r['latency_ms_mean']:.2f} ± {r['latency_ms_std']:.2f} ms")
    return results
