"""
FastAPI backend for ANN-Bench.

Endpoints:
  GET  /              → serves index.html
  GET  /api/data      → PCA 3D coords + labels + speaker_ids
  POST /api/query     → { speaker, method, k } → Top-K results + latency
  GET  /api/benchmark → full recall@k + latency comparison
  GET  /api/audio/{idx} → real .flac (LibriSpeech) or synthetic WAV fallback
"""

import io
import os
import time
import struct
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
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

USE_SYNTHETIC = os.environ.get("USE_SYNTHETIC", "0") == "1"
_ready = False
embeddings = labels = speaker_ids = audio_paths = None
HAS_REAL_AUDIO = False
_indexes = {}
_coords3d = []

@app.on_event("startup")
async def startup():
    global embeddings, labels, speaker_ids, audio_paths
    global HAS_REAL_AUDIO, _indexes, _coords3d, _ready

    print(f"Loading embeddings (synthetic={USE_SYNTHETIC})...")
    embeddings, labels, speaker_ids, audio_paths = load_embeddings(use_synthetic=USE_SYNTHETIC)

    HAS_REAL_AUDIO = any(p for p in audio_paths)
    print(f"Real audio available: {HAS_REAL_AUDIO}")

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

    _ready = True
    print(f"Ready — {len(embeddings)} embeddings, {len(set(labels))} speakers")

# ── Routes ────────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent.parent.parent / "static"


@app.get("/api/ready")
def ready():
    return {"ready": _ready}

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/data")
def get_data():
    import colorsys
    n_speakers = len(set(labels))

    def spk_hex(i):
        r, g, b = colorsys.hls_to_rgb(i / n_speakers, 0.55, 0.7)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    return {
        "coords": _coords3d,
        "labels": labels.tolist(),
        "speaker_ids": speaker_ids,
        "colors": [spk_hex(int(l)) for l in labels],
        "n_speakers": n_speakers,
        "has_real_audio": HAS_REAL_AUDIO,
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


@app.get("/api/audio/{embedding_idx}")
def get_audio(embedding_idx: int):
    """
    Serve audio for a given embedding index.
    - If real LibriSpeech .flac exists → stream it directly (audio/flac)
    - Otherwise → generate synthetic WAV (audio/wav)
    """
    if embedding_idx < 0 or embedding_idx >= len(audio_paths):
        raise HTTPException(status_code=404, detail="Index out of range")

    fpath = audio_paths[embedding_idx] if embedding_idx < len(audio_paths) else ""

    # ── Real audio ────────────────────────────────────────────────────────────
    if fpath and Path(fpath).exists():
        suffix = Path(fpath).suffix.lower()
        media = "audio/flac" if suffix == ".flac" else "audio/wav"
        return FileResponse(fpath, media_type=media)

    # ── Synthetic fallback ────────────────────────────────────────────────────
    spk_label = int(labels[embedding_idx])
    return _synth_wav(spk_label)


def _synth_wav(speaker_label: int) -> Response:
    sample_rate = 22050
    duration = 2.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    base_freq = 80 + (speaker_label * 37) % 200
    vibrato   = 1 + 0.008 * np.sin(2 * np.pi * 5 * t)

    signal = (
        0.50 * np.sin(2 * np.pi * base_freq * vibrato * t) +
        0.25 * np.sin(2 * np.pi * base_freq * 2 * vibrato * t) +
        0.12 * np.sin(2 * np.pi * base_freq * 3 * vibrato * t) +
        0.06 * np.sin(2 * np.pi * base_freq * 4 * vibrato * t)
    )

    env = np.ones(n_samples)
    fade = int(sample_rate * 0.05)
    env[:fade]  = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    signal = (signal * env * 0.7).astype(np.float32)

    pcm = (signal * 32767).astype(np.int16)
    buf = io.BytesIO()
    data_bytes = pcm.tobytes()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + len(data_bytes)))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate,
                          sample_rate * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', len(data_bytes)))
    buf.write(data_bytes)
    return Response(content=buf.getvalue(), media_type="audio/wav")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
