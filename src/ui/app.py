"""
Streamlit query interface for ANN-Bench.

Usage:
    streamlit run src/ui/app.py
"""

import time
import numpy as np
import streamlit as st
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
st.caption("Approximate Nearest Neighbor search — Flat Search vs KD-Tree vs LSH")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_data():
    return load_embeddings(use_synthetic=True)

@st.cache_resource
def get_indexes(_embeddings):
    flat   = FlatSearch();  flat.build(_embeddings)
    kdtree = KDTree();      kdtree.build(_embeddings)
    lsh    = LSH();         lsh.build(_embeddings)
    return {"Flat": flat, "KD-Tree": kdtree, "LSH": lsh}

@st.cache_resource
def get_pca_coords(_embeddings):
    pca = PCA(n_components=3, random_state=42)
    return pca.fit_transform(_embeddings)

embeddings, labels, speaker_ids, _audio_paths = get_data()
indexes = get_indexes(embeddings)
coords3d = get_pca_coords(embeddings)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Query Settings")
    method = st.selectbox("Search method", ["Flat", "KD-Tree", "LSH"])
    k = st.slider("Top-K neighbours", 1, 20, 10)

    speaker_options = sorted(set(speaker_ids))
    query_speaker = st.selectbox("Query speaker", speaker_options)
    speaker_label = speaker_ids.index(query_speaker)
    candidate_idxs = np.where(labels == speaker_label)[0]
    query_sample = st.slider("Sample index", 0, len(candidate_idxs) - 1, 0)
    query_idx = int(candidate_idxs[query_sample])

    run_benchmark = st.button("Run Full Benchmark", type="secondary")

# ── Query ─────────────────────────────────────────────────────────────────────
selected_index = indexes[method]
q = embeddings[query_idx]
t0 = time.perf_counter()
neighbor_indices, scores = selected_index.query(q, k=k)
latency_ms = (time.perf_counter() - t0) * 1000
neighbor_set = set(neighbor_indices.tolist())

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("3D Speaker Embedding Space (PCA)")

    # Build plotly 3D scatter
    n_speakers = len(set(labels))
    import colorsys
    def spk_color(i):
        r, g, b = colorsys.hls_to_rgb(i / n_speakers, 0.55, 0.7)
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    # Separate into: normal / neighbours / query
    mask_normal   = np.array([i not in neighbor_set and i != query_idx for i in range(len(embeddings))])
    mask_neighbor = np.array([i in neighbor_set and i != query_idx for i in range(len(embeddings))])

    traces = []

    # Normal points (grouped by speaker for legend)
    for spk_label in sorted(set(labels)):
        m = mask_normal & (labels == spk_label)
        if m.sum() == 0:
            continue
        traces.append(go.Scatter3d(
            x=coords3d[m, 0], y=coords3d[m, 1], z=coords3d[m, 2],
            mode="markers",
            marker=dict(size=3, color=spk_color(spk_label), opacity=0.5),
            name=speaker_ids[spk_label],
            showlegend=False,
        ))

    # Neighbours
    if mask_neighbor.sum() > 0:
        traces.append(go.Scatter3d(
            x=coords3d[mask_neighbor, 0],
            y=coords3d[mask_neighbor, 1],
            z=coords3d[mask_neighbor, 2],
            mode="markers",
            marker=dict(size=8, color="gold", opacity=1.0, symbol="diamond"),
            name="Neighbours",
        ))

    # Query point
    traces.append(go.Scatter3d(
        x=[coords3d[query_idx, 0]],
        y=[coords3d[query_idx, 1]],
        z=[coords3d[query_idx, 2]],
        mode="markers",
        marker=dict(size=12, color="white", opacity=1.0, symbol="cross"),
        name="Query",
    ))

    # Lines from query to neighbours
    line_traces = []
    for ni in neighbor_indices:
        line_traces.append(go.Scatter3d(
            x=[coords3d[query_idx, 0], coords3d[ni, 0]],
            y=[coords3d[query_idx, 1], coords3d[ni, 1]],
            z=[coords3d[query_idx, 2], coords3d[ni, 2]],
            mode="lines",
            line=dict(color="gold", width=2),
            showlegend=False,
            opacity=0.6,
        ))

    fig = go.Figure(data=traces + line_traces)
    fig.update_layout(
        height=520,
        scene=dict(
            bgcolor="#0a0a1a",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="white",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(color="white")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Three.js tab (extra)
    with st.expander("✨ Three.js version (experimental)", expanded=False):
        html = build_threejs_html(
            embeddings, labels, speaker_ids,
            query_idx=query_idx,
            neighbor_indices=neighbor_indices,
            height=480,
        )
        st.html(html)

with col2:
    st.subheader("Results")
    st.metric("Method", method)
    st.metric("Latency", f"{latency_ms:.2f} ms")
    st.metric("Candidates returned", len(neighbor_indices))

    st.markdown("**Top-K neighbours**")
    score_label = "dist" if method == "KD-Tree" else "sim"
    for rank, (ni, score) in enumerate(zip(neighbor_indices, scores), 1):
        spk = speaker_ids[labels[ni]]
        match = "✅" if spk == query_speaker else "❌"
        st.markdown(f"`#{rank}` {match} `{spk}` ({score_label}={score:.3f})")

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
    names = list(results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    with bcol1:
        recalls = [results[n]["recall_at_k"] for n in names]
        fig = go.Figure(go.Bar(
            x=names, y=recalls, marker_color=colors,
            text=[f"{r:.3f}" for r in recalls], textposition="outside",
        ))
        fig.update_layout(
            title=f"Recall@{k}", yaxis_range=[0, 1.15],
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="white", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with bcol2:
        latencies = [results[n]["latency_ms_mean"] for n in names]
        errors    = [results[n]["latency_ms_std"] for n in names]
        fig2 = go.Figure(go.Bar(
            x=names, y=latencies, marker_color=colors,
            error_y=dict(type="data", array=errors),
            text=[f"{v:.2f}" for v in latencies], textposition="outside",
        ))
        fig2.update_layout(
            title="Query Latency (ms)",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="white", height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)
