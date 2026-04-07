"""
Data loader for LibriSpeech speaker embeddings.

Pipeline:
  1. Download LibriSpeech train-clean-100 (if not cached)
  2. Extract speaker embeddings via pretrained ECAPA-TDNN (SpeechBrain)
  3. Save embeddings + labels as .npy for fast reuse

If SpeechBrain is unavailable, falls back to synthetic embeddings for development.
"""

import os
import numpy as np
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "data"
EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
LABELS_PATH = CACHE_DIR / "labels.npy"
SPEAKER_IDS_PATH = CACHE_DIR / "speaker_ids.npy"


def extract_embeddings_speechbrain(max_speakers: int = 50) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Download LibriSpeech train-clean-100 and extract speaker embeddings
    using pretrained ECAPA-TDNN from SpeechBrain.

    Returns:
        embeddings: (N, 192) float32 array
        labels:     (N,)     int array — speaker index (0-based)
        speaker_ids: list of original LibriSpeech speaker ID strings
    """
    import torchaudio
    from speechbrain.pretrained import EncoderClassifier

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ECAPA-TDNN model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(CACHE_DIR / "ecapa_model"),
    )

    print("Downloading LibriSpeech train-clean-100...")
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=str(CACHE_DIR),
        url="train-clean-100",
        download=True,
    )

    # Group by speaker
    speaker_to_indices: dict[int, list[int]] = {}
    for i, (_, _, _, speaker_id, *_) in enumerate(dataset):
        speaker_to_indices.setdefault(speaker_id, []).append(i)

    selected_speakers = sorted(speaker_to_indices.keys())[:max_speakers]
    print(f"Using {len(selected_speakers)} speakers")

    embeddings, labels = [], []
    speaker_ids = []

    for label_idx, speaker_id in enumerate(selected_speakers):
        speaker_ids.append(str(speaker_id))
        for idx in speaker_to_indices[speaker_id]:
            waveform, sample_rate, *_ = dataset[idx]
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            emb = classifier.encode_batch(waveform).squeeze().detach().numpy()
            embeddings.append(emb)
            labels.append(label_idx)

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return embeddings, labels, speaker_ids


def generate_synthetic_embeddings(
    n_speakers: int = 50,
    samples_per_speaker: int = 20,
    dim: int = 192,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Generate synthetic speaker embeddings for development/testing.
    Each speaker is a Gaussian cluster in embedding space.

    Returns:
        embeddings: (N, dim) float32
        labels:     (N,)     int
        speaker_ids: list of str
    """
    rng = np.random.default_rng(seed)
    embeddings, labels = [], []

    for i in range(n_speakers):
        center = rng.normal(0, 1, dim).astype(np.float32)
        center /= np.linalg.norm(center)
        cluster = center + rng.normal(0, 0.1, (samples_per_speaker, dim)).astype(np.float32)
        # L2-normalize each embedding
        norms = np.linalg.norm(cluster, axis=1, keepdims=True)
        cluster = cluster / norms
        embeddings.append(cluster)
        labels.extend([i] * samples_per_speaker)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels, dtype=np.int32)
    speaker_ids = [f"speaker_{i:04d}" for i in range(n_speakers)]
    return embeddings, labels, speaker_ids


def load_embeddings(use_synthetic: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load embeddings from cache, or extract/generate if not cached.

    Args:
        use_synthetic: force synthetic data (for development)

    Returns:
        embeddings: (N, D) float32
        labels:     (N,)   int
        speaker_ids: list of str
    """
    if not use_synthetic and EMBEDDINGS_PATH.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(EMBEDDINGS_PATH)
        labels = np.load(LABELS_PATH)
        speaker_ids = np.load(SPEAKER_IDS_PATH, allow_pickle=True).tolist()
        print(f"Loaded {len(embeddings)} embeddings, {len(set(labels))} speakers, dim={embeddings.shape[1]}")
        return embeddings, labels, speaker_ids

    if use_synthetic:
        print("Generating synthetic embeddings...")
        embeddings, labels, speaker_ids = generate_synthetic_embeddings()
    else:
        try:
            embeddings, labels, speaker_ids = extract_embeddings_speechbrain()
        except ImportError:
            print("SpeechBrain not available, falling back to synthetic embeddings.")
            embeddings, labels, speaker_ids = generate_synthetic_embeddings()

    # Cache to disk
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)
    np.save(SPEAKER_IDS_PATH, np.array(speaker_ids, dtype=object))
    print(f"Saved {len(embeddings)} embeddings → {CACHE_DIR}")

    return embeddings, labels, speaker_ids


if __name__ == "__main__":
    embs, lbls, ids = load_embeddings(use_synthetic=True)
    print(f"embeddings: {embs.shape}, labels: {lbls.shape}, speakers: {len(ids)}")
