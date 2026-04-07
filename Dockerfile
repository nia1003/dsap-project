FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir numpy matplotlib scikit-learn streamlit plotly

COPY . .

EXPOSE 8601

CMD ["python3", "-m", "streamlit", "run", "src/ui/app.py", \
     "--server.port=8601", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
