"""
Data loader for LibriSpeech speaker embeddings.

Pipeline:
  1. Download LibriSpeech test-clean (~350MB) manually via urllib if not cached
  2. Extract speaker embeddings via pretrained ECAPA-TDNN (SpeechBrain)
  3. Save embeddings + labels + audio_paths as .npy for fast reuse
"""

import json
import tarfile
import urllib.request
import numpy as np
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "data"
EMBEDDINGS_PATH  = CACHE_DIR / "embeddings.npy"
LABELS_PATH      = CACHE_DIR / "labels.npy"
SPEAKER_IDS_PATH = CACHE_DIR / "speaker_ids.npy"
AUDIO_PATHS_PATH = CACHE_DIR / "audio_paths.npy"
CACHE_META_PATH  = CACHE_DIR / "cache_meta.json"

REAL_CONFIG = {"subset": "test-clean", "max_speakers": 5, "max_utterances": 20}

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


def _cache_valid() -> bool:
    if not EMBEDDINGS_PATH.exists():
        return False
    if not CACHE_META_PATH.exists():
        return False
    try:
        meta = json.loads(CACHE_META_PATH.read_text())
        return meta == REAL_CONFIG
    except Exception:
        return False


def _ensure_librispeech() -> Path:
    """Download and extract LibriSpeech test-clean if not already present."""
    libri_root = CACHE_DIR / "LibriSpeech" / REAL_CONFIG["subset"]
    if libri_root.exists() and any(libri_root.rglob("*.flac")):
        print(f"LibriSpeech already downloaded at {libri_root}")
        return libri_root

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = CACHE_DIR / "test-clean.tar.gz"

    if not tar_path.exists():
        print(f"Downloading LibriSpeech test-clean (~350MB)...")
        def progress(count, block_size, total):
            pct = count * block_size * 100 // total
            if pct % 10 == 0:
                print(f"  {pct}%", flush=True)
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path, reporthook=progress)
        print("Download complete.")

    print("Extracting...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(CACHE_DIR)
    print("Extraction complete.")
    return libri_root


def extract_embeddings_speechbrain() -> tuple:
    """
    Download LibriSpeech test-clean and extract speaker embeddings for 5 speakers.
    Uses manual download + directory scanning — does not use torchaudio.datasets.
    """
    import torchaudio
    from speechbrain.inference import EncoderClassifier

    max_speakers = REAL_CONFIG["max_speakers"]
    max_utt      = REAL_CONFIG["max_utterances"]

    print("Loading ECAPA-TDNN model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(CACHE_DIR / "ecapa_model"),
    )

    libri_root = _ensure_librispeech()

    # Scan directory: LibriSpeech/test-clean/{speaker_id}/{chapter_id}/*.flac
    speaker_dirs = sorted([d for d in libri_root.iterdir() if d.is_dir()])[:max_speakers]
    print(f"Using {len(speaker_dirs)} speakers, up to {max_utt} utterances each")

    embeddings, labels, audio_paths = [], [], []
    speaker_ids = []

    for label_idx, spk_dir in enumerate(speaker_dirs):
        speaker_ids.append(spk_dir.name)
        flac_files = sorted(spk_dir.rglob("*.flac"))[:max_utt]
        extracted = 0
        for fpath in flac_files:
            try:
                waveform, sample_rate = torchaudio.load(str(fpath))
                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                # encode_batch expects (batch, samples); waveform is (channels, samples)
                if waveform.dim() == 2 and waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)  # stereo → mono
                emb = classifier.encode_batch(waveform).squeeze(0).detach().numpy()
                if emb.ndim == 0:
                    emb = emb.reshape(1)
                embeddings.append(emb)
                labels.append(label_idx)
                audio_paths.append(str(fpath))
                extracted += 1
            except Exception as e:
                print(f"    Skip {fpath.name}: {e}")
        print(f"  Speaker {spk_dir.name}: {extracted}/{len(flac_files)} utterances extracted")

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return embeddings, labels, speaker_ids, audio_paths


def generate_synthetic_embeddings(
    n_speakers: int = 50,
    samples_per_speaker: int = 20,
    dim: int = 192,
    seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)
    embeddings, labels = [], []

    for i in range(n_speakers):
        center = rng.normal(0, 1, dim).astype(np.float32)
        center /= np.linalg.norm(center)
        cluster = center + rng.normal(0, 0.1, (samples_per_speaker, dim)).astype(np.float32)
        norms = np.linalg.norm(cluster, axis=1, keepdims=True)
        cluster = cluster / norms
        embeddings.append(cluster)
        labels.extend([i] * samples_per_speaker)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels, dtype=np.int32)
    speaker_ids = [f"speaker_{i:04d}" for i in range(n_speakers)]
    audio_paths = [""] * len(embeddings)
    return embeddings, labels, speaker_ids, audio_paths


def load_embeddings(use_synthetic: bool = False) -> tuple:
    """
    Returns: (embeddings, labels, speaker_ids, audio_paths)
    """
    if not use_synthetic and _cache_valid():
        print("Loading cached embeddings...")
        embeddings  = np.load(EMBEDDINGS_PATH)
        labels      = np.load(LABELS_PATH)
        speaker_ids = np.load(SPEAKER_IDS_PATH, allow_pickle=True).tolist()
        audio_paths = np.load(AUDIO_PATHS_PATH, allow_pickle=True).tolist() \
                      if AUDIO_PATHS_PATH.exists() else [""] * len(embeddings)
        print(f"Loaded {len(embeddings)} embeddings, {len(set(labels))} speakers, dim={embeddings.shape[1]}")
        return embeddings, labels, speaker_ids, audio_paths

    if use_synthetic:
        print("Generating synthetic embeddings...")
        result = generate_synthetic_embeddings()
    else:
        try:
            result = extract_embeddings_speechbrain()
        except Exception as e:
            print(f"Real embedding extraction failed: {e}")
            print("Falling back to synthetic embeddings.")
            result = generate_synthetic_embeddings()

    embeddings, labels, speaker_ids, audio_paths = result

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)
    np.save(SPEAKER_IDS_PATH, np.array(speaker_ids, dtype=object))
    np.save(AUDIO_PATHS_PATH, np.array(audio_paths, dtype=object))
    if not use_synthetic:
        CACHE_META_PATH.write_text(json.dumps(REAL_CONFIG))
    print(f"Saved {len(embeddings)} embeddings → {CACHE_DIR}")

    return embeddings, labels, speaker_ids, audio_paths


if __name__ == "__main__":
    embs, lbls, ids, paths = load_embeddings(use_synthetic=False)
    print(f"embeddings: {embs.shape}, speakers: {ids}, dim={embs.shape[1]}")
    print(f"audio paths sample: {paths[:3]}")
