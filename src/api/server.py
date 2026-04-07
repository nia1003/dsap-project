"""
FastAPI backend for ANN-Bench.

Endpoints:
  GET  /              → serves index.html
  GET  /api/data      → PCA 3D coords + labels + speaker_ids
  POST /api/query     → { speaker, method, k } → Top-K results + latency
  GET  /api/benchmark → full recall@k + latency comparison
"""

import time
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.decomposition import PCA

from src.data.loader import load_embeddings
from src.index.flat import FlatSearch
from src.index.kdtree import KDTree
from src.index.lsh import LSH
from src.benchmark.eval import compare_all

app = FastAPI(title="ANN-Bench")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load data once ───────────────────────────────────────────────────

print("Loading embeddings...")
embeddings, labels, speaker_ids = load_embeddings(use_synthetic=True)

print("Building indexes...")
_indexes = {
    "Flat":   FlatSearch(),
    "KDTree": KDTree(),
    "LSH":    LSH(),
}
for idx in _indexes.values():
    idx.build(embeddings)

print("Computing PCA...")
_pca = PCA(n_components=3, random_state=42)
_coords3d = _pca.fit_transform(embeddings).tolist()

print(f"Ready — {len(embeddings)} embeddings, {len(set(labels))} speakers")

# ── Routes ────────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent.parent.parent / "static"

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/data")
def get_data():
    import colorsys
    n_speakers = len(set(labels))
    def spk_hex(i):
        r, g, b = colorsys.hls_to_rgb(i / n_speakers, 0.55, 0.7)
        return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    return {
        "coords": _coords3d,
        "labels": labels.tolist(),
        "speaker_ids": speaker_ids,
        "colors": [spk_hex(int(l)) for l in labels],
        "n_speakers": n_speakers,
    }


class QueryRequest(BaseModel):
    speaker: str
    method: str = "Flat"
    k: int = 10
    sample: int = 0


@app.post("/api/query")
def query(req: QueryRequest):
    spk_label = speaker_ids.index(req.speaker)
    candidate_idxs = np.where(labels == spk_label)[0]
    query_idx = int(candidate_idxs[req.sample % len(candidate_idxs)])

    idx_key = {"Flat": "Flat", "KD-Tree": "KDTree", "LSH": "LSH"}.get(req.method, "Flat")
    index = _indexes[idx_key]

    q = embeddings[query_idx]
    t0 = time.perf_counter()
    neighbor_indices, scores = index.query(q, k=req.k)
    latency_ms = (time.perf_counter() - t0) * 1000

    results = []
    for rank, (ni, score) in enumerate(zip(neighbor_indices.tolist(), scores.tolist()), 1):
        spk = speaker_ids[labels[ni]]
        results.append({
            "rank": rank,
            "index": ni,
            "speaker": spk,
            "score": round(float(score), 4),
            "match": spk == req.speaker,
        })

    return {
        "query_idx": query_idx,
        "neighbor_indices": neighbor_indices.tolist(),
        "results": results,
        "latency_ms": round(latency_ms, 3),
        "method": req.method,
        "k": req.k,
    }


@app.get("/api/benchmark")
def benchmark(k: int = 10, n_queries: int = 100):
    results = compare_all(
        {"Flat": FlatSearch(), "KD-Tree": KDTree(), "LSH": LSH()},
        embeddings, k=k, n_queries=n_queries,
    )
    return results


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
