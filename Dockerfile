FROM python:3.12-slim

# Avoid interactive prompts; keep Python output unbuffered for container logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# System deps: libpq for psycopg, build tools dropped (psycopg[binary] ships wheels).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Cache the embedding AND reranker models into the image's HF_HOME so the container
# runs fully air-gapped at runtime (no HuggingFace access needed). Done BEFORE copying
# source so editing code doesn't invalidate the (expensive) model layer.
ARG EMBEDDING_MODEL=BAAI/bge-m3
ARG RERANKER_MODEL=BAAI/bge-reranker-base
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); \
CrossEncoder('${RERANKER_MODEL}')"

# Force model loads to stay local at runtime. Set AFTER the bake above so the build
# can still download; these only affect subsequent layers and the running container.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# App source (changes here reuse all layers above).
COPY rag_lite ./rag_lite
COPY api ./api
COPY main.py setup.py ./

EXPOSE 8000

# Single worker: the embedding model + reranker are loaded once per process and
# are memory-heavy. Scale horizontally with more containers, not more workers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
