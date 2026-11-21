# Production image - used by Hugging Face Spaces (Docker SDK) and any Docker host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# CPU-only PyTorch keeps the image small (~1 GB instead of ~5 GB with CUDA)
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p outputs/results && chown -R appuser:appuser outputs
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",7860)}/api/health')"

CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
