# Maintenance Copilot API image (also used for the one-shot init job in docker compose).
FROM python:3.12-slim

# COPILOT_HOME: project folder (data/, models/, index/, knowledge/) for the installed package.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    COPILOT_HOME=/app EMBEDDING_CACHE_DIR=/app/.cache/fastembed
# libgomp1: OpenMP runtime for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY scripts ./scripts
COPY knowledge ./knowledge
# Download the embedding model and build the document index into the image.
RUN python scripts/build_index.py \
    && mkdir -p data models && chown -R app:app /app
COPY --chmod=755 docker/init.sh docker/init.sh

USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"
CMD ["uvicorn", "copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
