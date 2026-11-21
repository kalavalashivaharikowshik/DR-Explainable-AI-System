import importlib
import io
import os

import cv2
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from tests.conftest import FakePredictor, synthetic_fundus


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("DR_REFERENCE_DIR", str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("DR_SKIP_MODEL_WARMUP", "1")
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)

    from src.config import config
    importlib.reload(config)
    import src.visualization.comparison_panel as cp
    importlib.reload(cp)
    import src.inference.pipeline as pipeline
    importlib.reload(pipeline)
    import app.backend.main as main
    importlib.reload(main)

    fake = FakePredictor()
    monkeypatch.setattr(pipeline, "get_predictor", lambda: fake)
    monkeypatch.setattr(main, "get_predictor", lambda: fake)

    with TestClient(main.app) as c:
        yield c


def jpeg_bytes(img=None):
    ok, buf = cv2.imencode(".jpg", synthetic_fundus() if img is None else img)
    return buf.tobytes()


def test_app_starts_without_data_folder_and_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "DR Explainable AI" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_config_and_health(client):
    cfg = client.get("/api/config").json()
    assert cfg["email_enabled"] is False
    assert cfg["reference_counts"] == [0, 0, 0, 0, 0]
    assert client.get("/api/health").json()["model_loaded"] is True


def test_full_analysis_flow(client):
    r = client.post("/api/analyze", files={"file": ("eye.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["prediction"] == "Moderate"
    assert len(d["probabilities"]) == 5
    for url in d["images"].values():
        img = client.get(url)
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"

    pdf = client.get(d["report_url"])
    assert pdf.status_code == 200
    reader = PdfReader(io.BytesIO(pdf.content))
    assert reader.is_encrypted
    assert not reader.decrypt("wrong-password")
    assert reader.decrypt(d["report_password"])


def test_password_not_stored_and_no_plain_pdf(client, tmp_path):
    d = client.post("/api/analyze", files={"file": ("eye.jpg", jpeg_bytes())}).json()
    folder = tmp_path / "results" / d["result_id"]
    files = os.listdir(folder)
    assert "report.pdf" not in files and "report_secure.pdf" in files
    for name in files:
        assert d["report_password"].encode() not in (folder / name).read_bytes()


def test_upload_filename_cannot_escape(client, tmp_path):
    target = tmp_path / "escaped.jpg"
    r = client.post("/api/analyze", files={"file": (str(target), jpeg_bytes())})
    assert r.status_code == 200
    assert not target.exists()


def test_rejects_bad_uploads(client):
    assert client.post("/api/analyze", files={"file": ("x.py", b"print(1)")}).status_code == 400
    assert client.post("/api/analyze", files={"file": ("x.jpg", b"not an image")}).status_code == 400
    assert client.post("/api/analyze", files={"file": ("x.jpg", b"")}).status_code == 400
    assert client.post("/api/analyze").status_code == 422


def test_result_paths_are_validated(client):
    assert client.get("/api/results/../../etc/passwd").status_code == 404
    assert client.get("/api/results/" + "a" * 32 + "/..%2F..%2Fsecret.jpg").status_code == 404
    assert client.get("/api/results/" + "a" * 32 + "/gradcam.jpg").status_code == 404
    assert client.get("/api/reports/not-an-id/report.pdf").status_code == 404


def test_old_insecure_endpoints_are_gone(client):
    assert client.post("/send-report", json={"email": "a@b.co", "pdf_path": "/etc/passwd"}).status_code in (404, 405)
    assert client.get("/outputs/report/dr_report.pdf").status_code == 404


def test_email_disabled_and_no_path_parameter(client):
    d = client.post("/api/analyze", files={"file": ("eye.jpg", jpeg_bytes())}).json()
    r = client.post(f"/api/reports/{d['result_id']}/email", json={"email": "doc@example.com", "pdf_path": "/etc/passwd"})
    assert r.status_code == 503


def test_email_sends_server_side_file(client, monkeypatch):
    import app.backend.main as main
    sent = []
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setattr(main, "send_email", lambda to, path, rid: sent.append((to, path)))

    d = client.post("/api/analyze", files={"file": ("eye.jpg", jpeg_bytes())}).json()
    assert client.post(f"/api/reports/{d['result_id']}/email", json={"email": "bad"}).status_code == 400
    r = client.post(f"/api/reports/{d['result_id']}/email", json={"email": "doc@example.com"})
    assert r.status_code == 200
    assert sent[0][0] == "doc@example.com" and sent[0][1].endswith(os.path.join(d["result_id"], "report_secure.pdf"))


def test_reference_endpoints_without_images(client):
    assert client.get("/api/reference/0").json()["images"] == []
    assert client.get("/api/reference/9").status_code == 404
    assert client.get("/api/reference/0/..%2F..%2Fsrc").status_code == 404


def test_rejects_decompression_bomb_and_downscales_large_images():
    import numpy as np
    from PIL import Image

    from src.inference.pipeline import InvalidImage, decode_image

    buf = io.BytesIO()
    Image.new("L", (9000, 6000)).save(buf, format="PNG")      # 54 MP, tiny file
    with pytest.raises(InvalidImage):
        decode_image(buf.getvalue())

    big = np.zeros((4000, 5000, 3), np.uint8)
    big[1000:3000, 1000:4000] = 120
    ok, enc = cv2.imencode(".jpg", big)
    assert max(decode_image(enc.tobytes()).shape[:2]) == 3072
