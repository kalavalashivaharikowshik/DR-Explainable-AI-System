"""FastAPI backend for the DR Explainable AI System.

Run locally:   uvicorn app.backend.main:app --reload
"""

import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import config
from src.email_service.send_email import EmailNotConfigured, is_valid_email, send_email
from src.inference import pipeline
from src.inference.predict import ModelNotAvailable, get_predictor
from src.utils.helpers import cleanup_old_results, is_safe_filename, is_valid_result_id
from src.utils.logger import get_logger
from src.visualization.comparison_panel import list_reference_images

log = get_logger("dr_xai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    removed = cleanup_old_results(config.RESULTS_DIR, config.RESULT_TTL_HOURS)
    if removed:
        log.info("Removed %d expired result folders", removed)
    if os.environ.get("DR_SKIP_MODEL_WARMUP") != "1":
        try:
            predictor = get_predictor()   # load the model once, at startup
            path = config.ONNX_MODEL_PATH if predictor.backend == "onnx" else config.MODEL_PATH
            log.info("Model loaded (%s backend) from %s", predictor.backend, path)
        except ModelNotAvailable as e:
            log.error("%s", e)
    yield


app = FastAPI(title="DR Explainable AI System", version="2.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")


# ------------------------------------------------------------------
# Pages & meta
# ------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(config.FRONTEND_DIR, "index.html"))


@app.get("/api/health")
def health():
    try:
        backend = get_predictor().backend
    except ModelNotAvailable:
        backend = None
    return {"status": "ok" if backend else "model_unavailable", "model_loaded": backend is not None,
            "backend": backend}


@app.get("/api/config")
def frontend_config():
    return {
        "model": config.MODEL_NAME,
        "classes": config.CLASS_NAMES,
        "email_enabled": config.email_enabled(),
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "result_ttl_hours": config.RESULT_TTL_HOURS,
        "reference_counts": [len(list_reference_images(i)) for i in range(config.NUM_CLASSES)],
    }


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------
def _parse_preprocessed_flag(value):
    if value in (None, "", "auto"):
        return None
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise HTTPException(400, "preprocessed must be 'auto', 'true' or 'false'.")


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), preprocessed: str = Form("auto")):
    # NOTE: plain `def` -> FastAPI runs it in a worker thread, so a slow analysis
    # never blocks other requests.
    name = os.path.basename(file.filename or "")
    ext = os.path.splitext(name)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}")

    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")

    flag = _parse_preprocessed_flag(preprocessed)

    cleanup_old_results(config.RESULTS_DIR, config.RESULT_TTL_HOURS)

    try:
        # the upload is processed in memory; the original file is never written to disk
        return pipeline.run_full_analysis(data, already_preprocessed=flag, results_dir=config.RESULTS_DIR)
    except pipeline.InvalidImage as e:
        raise HTTPException(400, str(e))
    except ModelNotAvailable as e:
        log.error("%s", e)
        raise HTTPException(503, "The AI model is not available on this server.")


# ------------------------------------------------------------------
# Result files (served by unguessable result id; no directory listing)
# ------------------------------------------------------------------
def _result_file(result_id: str, filename: str) -> str:
    if not (is_valid_result_id(result_id) and is_safe_filename(filename)):
        raise HTTPException(404, "Not found")
    path = os.path.join(config.RESULTS_DIR, result_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Result not found or expired.")
    return path


@app.get("/api/results/{result_id}/{filename}")
def result_image(result_id: str, filename: str):
    if not filename.endswith(".jpg"):
        raise HTTPException(404, "Not found")
    return FileResponse(_result_file(result_id, filename), media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/reports/{result_id}/report.pdf")
def report_pdf(result_id: str, download: bool = False):
    path = _result_file(result_id, "report_secure.pdf")
    disposition = "attachment" if download else "inline"
    return FileResponse(path, media_type="application/pdf",
                        headers={"Content-Disposition": f'{disposition}; filename="DR_AI_Report_{result_id[:8]}.pdf"',
                                 "Cache-Control": "no-store"})


# ------------------------------------------------------------------
# Email (server decides which file is sent; the client only names a result id)
# ------------------------------------------------------------------
class EmailRequest(BaseModel):
    email: str


_email_log = defaultdict(deque)
_email_lock = threading.Lock()


def _rate_limited(client: str) -> bool:
    now = time.time()
    with _email_lock:
        q = _email_log[client]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= config.EMAILS_PER_HOUR_PER_CLIENT:
            return True
        q.append(now)
        return False


@app.post("/api/reports/{result_id}/email")
def email_report(result_id: str, body: EmailRequest, request: Request):
    if not config.email_enabled():
        raise HTTPException(503, "Email is not configured on this server.")

    address = body.email.strip()
    if not is_valid_email(address):
        raise HTTPException(400, "Please enter a valid email address.")

    path = _result_file(result_id, "report_secure.pdf")

    client = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if _rate_limited(client):
        raise HTTPException(429, "Too many emails sent. Please try again later.")

    try:
        send_email(address, path, result_id)
    except EmailNotConfigured as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # SMTP failures, timeouts...
        log.error("Email sending failed: %s", e)
        raise HTTPException(502, "The email could not be sent. Please try again later.")

    return {"status": "sent"}


# ------------------------------------------------------------------
# Reference cases
# ------------------------------------------------------------------
@app.get("/api/reference/{class_index}")
def reference_list(class_index: int):
    if not 0 <= class_index < config.NUM_CLASSES:
        raise HTTPException(404, "Unknown class")
    files = list_reference_images(class_index)
    return {"class": config.CLASS_NAMES[class_index],
            "images": [f"/api/reference/{class_index}/{f}" for f in files]}


@app.get("/api/reference/{class_index}/{filename}")
def reference_image(class_index: int, filename: str):
    if not 0 <= class_index < config.NUM_CLASSES or filename not in list_reference_images(class_index):
        raise HTTPException(404, "Not found")
    return FileResponse(os.path.join(config.REFERENCE_DIR, str(class_index), filename),
                        headers={"Cache-Control": "public, max-age=86400"})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
