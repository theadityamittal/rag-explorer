# ---------- Build image ----------
FROM python:3.11-slim

# System deps (optional but handy for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirement files first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY src /app/src
COPY docs /app/docs
COPY configs /app/configs
COPY data /app/data

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Note: we rely on Ollama running outside the container by default.
# If Ollama is on host, pass OLLAMA_HOST at runtime, e.g.:
# -e OLLAMA_HOST=http://host.docker.internal:11434

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
