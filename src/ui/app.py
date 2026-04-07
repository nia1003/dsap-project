"""
Streamlit query interface for ANN-Bench.

Usage:
    streamlit run src/ui/app.py
"""

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.data.loader import load_embeddings
from src.index.flat import FlatSearch
from src.index.kdtree import KDTree
from src.index.lsh import LSH
from src.benchmark.eval import compare_all
from src.ui.threejs_component import build_threejs_html

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANN-Bench: Speaker Search",
    page_icon="🔊",
    layout="wide",
)

st.title("🔊 ANN-Bench: Speaker Embedding Search")
st.caption("Approximate Nearest Neighbor search across Flat Search, KD-Tree, and LSH")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_data():
    return load_embeddings(use_synthetic=True)

@st.cache_resource
def get_indexes(embeddings):
    flat   = FlatSearch();  flat.build(embeddings)
    kdtree = KDTree();      kdtree.build(embeddings)
    lsh    = LSH();         lsh.build(embeddings)
    return {"Flat": flat, "KD-Tree": kdtree, "LSH": lsh}

embeddings, labels, speaker_ids = get_data()
indexes = get_indexes(embeddings)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Query Settings")

    method = st.selectbox("Search method", ["Flat", "KD-Tree", "LSH"])
    k = st.slider("Top-K neighbours", 1, 20, 10)

    speaker_options = sorted(set(speaker_ids))
    query_speaker = st.selectbox("Query speaker", speaker_options)

    # Pick a random embedding from the selected speaker
    speaker_label = speaker_ids.index(query_speaker)
    candidate_idxs = np.where(labels == speaker_label)[0]
    query_sample = st.slider(
        "Sample index", 0, len(candidate_idxs) - 1, 0,
        help="Pick which utterance from this speaker to use as query"
    )
    query_idx = int(candidate_idxs[query_sample])

    run_benchmark = st.button("Run Full Benchmark", type="secondary")

# ── Query ─────────────────────────────────────────────────────────────────────
idx_map = {"Flat": indexes["Flat"], "KD-Tree": indexes["KD-Tree"], "LSH": indexes["LSH"]}
selected_index = idx_map[method]

import time
q = embeddings[query_idx]
t0 = time.perf_counter()
neighbor_indices, scores = selected_index.query(q, k=k)
latency_ms = (time.perf_counter() - t0) * 1000

# ── Layout: 3D viz + results ──────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("3D Speaker Embedding Space")
    st.caption("⬜ Query  🟡 Neighbours  🔵 Others — drag to rotate, scroll to zoom")
    html = build_threejs_html(
        embeddings, labels, speaker_ids,
        query_idx=query_idx,
        neighbor_indices=neighbor_indices,
        height=520,
    )
    components.html(html, height=530, scrolling=False)

with col2:
    st.subheader("Results")
    st.metric("Method", method)
    st.metric("Latency", f"{latency_ms:.2f} ms")
    st.metric("Candidates returned", len(neighbor_indices))

    st.markdown("**Top-K neighbours**")
    for rank, (ni, score) in enumerate(zip(neighbor_indices, scores), 1):
        spk = speaker_ids[labels[ni]]
        match = "✅" if spk == query_speaker else "❌"
        st.markdown(f"`#{rank}` {match} `{spk}` (sim={score:.3f})")

# ── Benchmark panel ───────────────────────────────────────────────────────────
if run_benchmark:
    st.divider()
    st.subheader("Full Benchmark")

    with st.spinner("Running benchmark (100 queries per method)..."):
        results = compare_all(
            {"Flat": FlatSearch(), "KD-Tree": KDTree(), "LSH": LSH()},
            embeddings, k=k, n_queries=100,
        )

    bcol1, bcol2 = st.columns(2)

    with bcol1:
        names = list(results.keys())
        recalls = [results[n]["recall_at_k"] for n in names]
        fig = go.Figure(go.Bar(
            x=names, y=recalls,
            marker_color=["#4C72B0", "#DD8452", "#55A868"],
            text=[f"{r:.3f}" for r in recalls], textposition="outside",
        ))
        fig.update_layout(
            title=f"Recall@{k}", yaxis_range=[0, 1.1],
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with bcol2:
        latencies = [results[n]["latency_ms_mean"] for n in names]
        errors    = [results[n]["latency_ms_std"] for n in names]
        fig2 = go.Figure(go.Bar(
            x=names, y=latencies,
            error_y=dict(type="data", array=errors),
            marker_color=["#4C72B0", "#DD8452", "#55A868"],
            text=[f"{v:.2f}" for v in latencies], textposition="outside",
        ))
        fig2.update_layout(
            title="Query Latency (ms)",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)
