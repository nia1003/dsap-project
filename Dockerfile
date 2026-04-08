FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    numpy matplotlib scikit-learn \
    fastapi "uvicorn[standard]"

RUN pip install --no-cache-dir speechbrain

RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8601

ENV USE_SYNTHETIC=0
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8601"]
