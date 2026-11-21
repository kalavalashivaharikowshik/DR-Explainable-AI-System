# Light production image: ONNX Runtime instead of PyTorch (~350 MB RAM in use),
# so it runs on free 512 MB tiers such as Render. Works on any Docker host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MALLOC_ARENA_MAX=2 \
    DR_BACKEND=onnx \
    PORT=10000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd -m -u 1000 appuser
COPY --chown=appuser:appuser . .
# download the real model if the build context only contains the Git LFS pointer
RUN python scripts/fetch_model.py \
 && rm -f models/trained/*.pth \
 && mkdir -p outputs/results && chown -R appuser:appuser outputs models
USER appuser

EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",10000)}/api/health')"

CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
