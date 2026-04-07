FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir numpy matplotlib scikit-learn streamlit plotly fastapi uvicorn[standard]

COPY . .

EXPOSE 8601

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8601"]
