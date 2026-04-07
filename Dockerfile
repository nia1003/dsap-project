FROM python:3.11-slim

# System deps for torchaudio .flac support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy matplotlib scikit-learn \
    fastapi "uvicorn[standard]" \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    speechbrain

COPY . .

EXPOSE 8601

# USE_SYNTHETIC=0 → auto-download LibriSpeech on first run
# USE_SYNTHETIC=1 → skip download, use generated embeddings
ENV USE_SYNTHETIC=0

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8601"]
