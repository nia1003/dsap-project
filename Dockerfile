FROM python:3.11-slim

ARG WITH_REAL_AUDIO=0

# Only needed for real LibriSpeech (.flac decoding)
RUN if [ "$WITH_REAL_AUDIO" = "1" ]; then \
    apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

COPY requirements.txt .

# Lightweight deps (always installed, ~30s)
RUN pip install --no-cache-dir \
    numpy matplotlib scikit-learn \
    fastapi "uvicorn[standard]"

# Heavy deps only for real LibriSpeech mode (~15 min)
RUN if [ "$WITH_REAL_AUDIO" = "1" ]; then \
    pip install --no-cache-dir speechbrain && \
    pip install --no-cache-dir torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY . .

EXPOSE 8601

ENV USE_SYNTHETIC=1
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8601"]
