import os
import re
import secrets
import shutil
import time

RESULT_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.(jpg|png|pdf)$")


def new_result_id() -> str:
    """Unguessable 128-bit id used as the folder name for one analysis."""
    return secrets.token_hex(16)


def is_valid_result_id(value: str) -> bool:
    return bool(RESULT_ID_RE.match(value or ""))


def is_safe_filename(value: str) -> bool:
    return bool(SAFE_NAME_RE.match(value or ""))


def cleanup_old_results(results_dir: str, max_age_hours: int) -> int:
    """Delete analysis folders older than max_age_hours. Returns how many were removed."""
    if not os.path.isdir(results_dir):
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for name in os.listdir(results_dir):
        path = os.path.join(results_dir, name)
        if is_valid_result_id(name) and os.path.isdir(path) and os.path.getmtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed
