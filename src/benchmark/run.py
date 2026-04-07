"""
Run full benchmark comparison across Flat Search, KD-Tree, and LSH.

Usage:
    python -m src.benchmark.run              # real embeddings (requires SpeechBrain)
    python -m src.benchmark.run --synthetic  # synthetic embeddings (no dependencies)
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import load_embeddings
from src.index.flat import FlatSearch
from src.index.kdtree import KDTree
from src.index.lsh import LSH
from src.benchmark.eval import compare_all


def run(use_synthetic: bool = False, k: int = 10, n_queries: int = 100):
    embeddings, labels, speaker_ids, _ = load_embeddings(use_synthetic=use_synthetic)
    print(f"\nDatabase: {len(embeddings)} embeddings | dim={embeddings.shape[1]} | "
          f"speakers={len(set(labels))}\n")

    indexes = {
        "Flat":   FlatSearch(),
        "KDTree": KDTree(leaf_size=10),
        "LSH":    LSH(n_bits=4, n_tables=16),
    }

    results = compare_all(indexes, embeddings, k=k, n_queries=n_queries)

    print("\n=== Summary ===")
    print(f"{'Method':<10} {'Recall@'+str(k):<12} {'Latency (ms)':<15}")
    print("-" * 37)
    for name, r in results.items():
        print(f"{name:<10} {r['recall_at_k']:.3f}        "
              f"{r['latency_ms_mean']:.2f} ± {r['latency_ms_std']:.2f}")

    _plot(results, k)
    return results


def _plot(results: dict, k: int):
    names = list(results.keys())
    recalls = [results[n]["recall_at_k"] for n in names]
    latencies = [results[n]["latency_ms_mean"] for n in names]
    errors = [results[n]["latency_ms_std"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    ax1.bar(names, recalls, color=colors)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel(f"Recall@{k}")
    ax1.set_title(f"Recall@{k} by Method")
    for i, v in enumerate(recalls):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    ax2.bar(names, latencies, yerr=errors, color=colors, capsize=5)
    ax2.set_ylabel("Query Latency (ms)")
    ax2.set_title("Average Query Latency")
    for i, v in enumerate(latencies):
        ax2.text(i, v + errors[i] + 0.01, f"{v:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150)
    print("\nPlot saved → benchmark_results.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic embeddings (no SpeechBrain required)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=100)
    args = parser.parse_args()

    run(use_synthetic=args.synthetic, k=args.k, n_queries=args.n_queries)
