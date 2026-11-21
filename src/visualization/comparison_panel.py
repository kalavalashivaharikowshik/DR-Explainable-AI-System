"""Reference cases for side-by-side comparison.

Put example images in data/reference_images/<class index>/ (0 = No DR ... 4 =
Proliferative). If the folder is missing the feature is simply disabled.
"""

import os
import random

from src.config.config import ALLOWED_EXTENSIONS, CLASS_NAMES, REFERENCE_DIR


def list_reference_images(class_index: int):
    folder = os.path.join(REFERENCE_DIR, str(class_index))
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS and not f.startswith(".")
    )


def reference_url(class_index: int, filename: str) -> str:
    return f"/api/reference/{class_index}/{filename}"


def get_reference_images(prediction: str):
    """One random example of the predicted grade and one of Proliferative DR (or None)."""
    class_index = CLASS_NAMES.index(prediction)

    def pick(idx):
        files = list_reference_images(idx)
        return reference_url(idx, random.choice(files)) if files else None

    return {
        "same_class_example": pick(class_index),
        "severe_example": pick(len(CLASS_NAMES) - 1),
    }
