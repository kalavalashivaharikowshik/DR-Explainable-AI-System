"""End-to-end check of a running deployment.

    python scripts/smoke_test.py https://dr-explainable-ai-ab6k.onrender.com

Uses a synthetic retina image (no patient data). Exits non-zero on any failure.
"""

import json
import sys
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def request(path, data=None, headers=None, timeout=180):
    req = urllib.request.Request(BASE + path, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def multipart(filename, content, fields=None):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in (fields or {}).items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def synthetic_retina():
    img = np.zeros((1400, 1600, 3), np.uint8)
    c = (800, 700)
    cv2.circle(img, c, 640, (40, 90, 200), -1)
    cv2.circle(img, (1080, 700), 70, (180, 220, 250), -1)
    for k in range(6):
        cv2.line(img, (1080, 700), (300 + 180 * k, 150 + 200 * k), (20, 40, 120), 6)
    for x, y in [(650, 760), (720, 560), (860, 900)]:
        cv2.circle(img, (x, y), 10, (150, 230, 255), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def main():
    print(f"Testing {BASE}\n")

    # wake up a sleeping free instance
    t = time.time()
    status, _, body = request("/api/health", timeout=180)
    health = json.loads(body) if status == 200 else {}
    check("health endpoint", status == 200 and health.get("model_loaded"), f"{health} in {time.time() - t:.1f}s")

    status, _, body = request("/")
    check("home page", status == 200 and b"DR Explainable AI" in body)
    for asset in ("/static/app.js", "/static/styles.css"):
        status, _, _ = request(asset)
        check(f"asset {asset}", status == 200)

    status, _, body = request("/api/config")
    config = json.loads(body) if status == 200 else {}
    check("config endpoint", status == 200 and len(config.get("classes", [])) == 5)

    image = synthetic_retina()
    result = None
    for attempt in (1, 2):
        data, headers = multipart("retina.jpg", image)
        t = time.time()
        status, _, body = request("/api/analyze", data, headers)
        elapsed = time.time() - t
        if status == 200:
            result = json.loads(body)
        check(f"analyze #{attempt}", status == 200 and result and result.get("prediction"),
              f"{result.get('prediction') if result else body[:120]} in {elapsed:.1f}s")

    if result:
        check("probabilities sum to 1", abs(sum(result["probabilities"]) - 1) < 1e-3)
        for key, url in result["images"].items():
            status, headers, body = request(url)
            check(f"image {key}", status == 200 and body[:2] == b"\xff\xd8", f"{len(body) // 1024} KB")
        status, headers, pdf = request(result["report_url"])
        check("encrypted PDF report", status == 200 and pdf.startswith(b"%PDF") and b"/Encrypt" in pdf,
              f"{len(pdf) // 1024} KB")
        check("report password returned", bool(result.get("report_password")))
        status, _, _ = request(f"/api/reports/{result['result_id']}/email",
                               json.dumps({"email": "doctor@example.com"}).encode(),
                               {"Content-Type": "application/json"})
        expected = 200 if config.get("email_enabled") else 503
        check("email endpoint behaves as configured", status in (expected, 429, 502), f"HTTP {status}")

    data, headers = multipart("notes.jpg", b"this is not an image")
    status, _, _ = request("/api/analyze", data, headers)
    check("rejects non-image upload", status == 400, f"HTTP {status}")

    data, headers = multipart("../../evil.py", b"print(1)")
    status, _, _ = request("/api/analyze", data, headers)
    check("rejects unsafe file type / path", status == 400, f"HTTP {status}")

    status, _, _ = request("/send-report", json.dumps({"email": "a@b.co", "pdf_path": "/etc/passwd"}).encode(),
                           {"Content-Type": "application/json"})
    check("old file-read endpoint removed", status in (404, 405), f"HTTP {status}")

    status, _, _ = request("/api/results/../../etc/passwd")
    check("path traversal blocked", status == 404, f"HTTP {status}")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
