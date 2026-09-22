# The AVM as a service.
#
#   docker build -t ames-avm .
#   docker run --rm -p 7860:7860 ames-avm
#   open http://localhost:7860/docs
#
# Two stages: dependencies resolve in the builder, only the installed environment and
# the promoted artifacts land in the final image.  Nothing from notebooks/, archive/,
# data/ or reports/ is copied in -- see .dockerignore.

FROM python:3.12-slim AS builder

# libgomp is LightGBM's OpenMP runtime.  Without it the model unpickles and then
# segfaults on the first predict, which is a memorable way to learn this.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serving.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements-serving.txt


FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=app:app src/ames/ ./src/ames/
COPY --chown=app:app serving/artifacts/ ./serving/artifacts/

USER app

# Hugging Face Spaces routes to 7860; anything else can override with -e PORT.
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,os,sys; \
      sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn ames.service:app --host 0.0.0.0 --port ${PORT}"]
