"""Central configuration.

Every setting can be overridden with an environment variable so the same code
runs locally, in Docker and on Hugging Face Spaces. Secrets (SMTP credentials)
are only ever read from the environment - never hard-code them.
"""

import os

from src.config.paths import PROJECT_ROOT


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_path(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return value if os.path.isabs(value) else os.path.join(PROJECT_ROOT, value)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
MODEL_PATH = _env_path("DR_MODEL_PATH", "models/trained/best_model.pth")
MODEL_NAME = "EfficientNet-B3"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
NUM_CLASSES = len(CLASS_NAMES)

# Must match the training pipeline (src/preprocessing + dataset_loader)
MODEL_INPUT_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Size of the images shown in the dashboard / PDF report
DISPLAY_SIZE = _env_int("DR_DISPLAY_SIZE", 512)

# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------
RESULTS_DIR = _env_path("DR_RESULTS_DIR", "outputs/results")
REFERENCE_DIR = _env_path("DR_REFERENCE_DIR", "data/reference_images")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "app", "frontend")

# Analysis results (images + PDF) are deleted after this many hours
RESULT_TTL_HOURS = _env_int("DR_RESULT_TTL_HOURS", 24)

# ------------------------------------------------------------------
# Uploads
# ------------------------------------------------------------------
MAX_UPLOAD_MB = _env_int("DR_MAX_UPLOAD_MB", 15)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ------------------------------------------------------------------
# Email (optional). Leave SMTP_USER / SMTP_PASSWORD unset to disable.
# ------------------------------------------------------------------
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = _env_int("SMTP_PORT", 465)
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_SENDER = os.environ.get("SMTP_SENDER", SMTP_USER)
EMAILS_PER_HOUR_PER_CLIENT = _env_int("DR_EMAILS_PER_HOUR", 5)


def email_enabled() -> bool:
    return bool(os.environ.get("SMTP_USER", SMTP_USER) and os.environ.get("SMTP_PASSWORD", SMTP_PASSWORD))
