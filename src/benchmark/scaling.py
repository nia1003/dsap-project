"""
Scaling benchmark: how recall@k and query latency change with database size.

Generates results for Flat / KDTree / LSH across increasing N,
then saves a two-panel plot (recall vs N, latency vs N).

Usage:
    python -m src.benchmark.scaling
    python -m src.benchmark.scaling --synthetic
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import load_embeddings
from src.index.flat import FlatSearch
from src.index.kdtree import KDTree
from src.index.lsh import LSH
from src.benchmark.eval import benchmark

DB_SIZES = [100, 250, 500, 1000, 2000]
K = 10
N_QUERIES = 50


def run_scaling(use_synthetic: bool = False):
    embeddings, _, _ = load_embeddings(use_synthetic=use_synthetic)
    max_n = min(len(embeddings), max(DB_SIZES))

    results = {name: {"recall": [], "latency": []}
               for name in ["Flat", "KDTree", "LSH"]}

    for n in DB_SIZES:
        if n > len(embeddings):
            break
        subset = embeddings[:n]
        print(f"\n--- N={n} ---")

        for name, idx in [("Flat", FlatSearch()),
                           ("KDTree", KDTree(leaf_size=10)),
                           ("LSH", LSH(n_bits=4, n_tables=16))]:
            r = benchmark(idx, subset, k=K, n_queries=min(N_QUERIES, n // 2))
            results[name]["recall"].append(r["recall_at_k"])
            results[name]["latency"].append(r["latency_ms_mean"])
            print(f"  {name:<8} recall={r['recall_at_k']:.3f}  "
                  f"latency={r['latency_ms_mean']:.3f}ms")

    _plot(results, DB_SIZES[:len(results["Flat"]["recall"])])
    return results


def _plot(results: dict, sizes: list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"Flat": "#4C72B0", "KDTree": "#DD8452", "LSH": "#55A868"}
    markers = {"Flat": "o", "KDTree": "s", "LSH": "^"}

    for name, data in results.items():
        n = len(data["recall"])
        ax1.plot(sizes[:n], data["recall"], marker=markers[name],
                 color=colors[name], label=name, linewidth=2)
        ax2.plot(sizes[:n], data["latency"], marker=markers[name],
                 color=colors[name], label=name, linewidth=2)

    ax1.set_xlabel("Database Size (N)")
    ax1.set_ylabel(f"Recall@{K}")
    ax1.set_title(f"Recall@{K} vs Database Size")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Database Size (N)")
    ax2.set_ylabel("Query Latency (ms)")
    ax2.set_title("Query Latency vs Database Size")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("scaling_results.png", dpi=150)
    print("\nPlot saved → scaling_results.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    run_scaling(use_synthetic=args.synthetic)
