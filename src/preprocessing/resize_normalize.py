"""Single source of truth for fundus preprocessing.

The exact same steps are used to build the training set
(src/preprocessing/run_preprocessing.py) and to prepare uploaded images at
inference time, so the model always sees images like the ones it was trained on.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocessing.ben_graham_preprocess import ben_graham_preprocess
from src.preprocessing.clahe_enhancement import apply_clahe
from src.preprocessing.crop_black_borders import crop_black_borders, field_of_view_mask


def preprocess_for_training(img: np.ndarray, size: int) -> np.ndarray:
    """crop -> CLAHE -> Ben Graham -> resize  (what the model was trained on)."""
    img = crop_black_borders(img)
    img = apply_clahe(img)
    img = ben_graham_preprocess(img)
    # INTER_LINEAR (OpenCV default) - the interpolation the training set was built with
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def looks_preprocessed(img: np.ndarray) -> bool:
    """Heuristic: is this image already Ben-Graham processed?

    Processed images have a neutral grey (~128) background and low colour
    saturation, while raw fundus photos have black corners and a strongly
    orange/red retina.
    """
    h, w = img.shape[:2]
    s = max(4, min(h, w) // 20)
    corners = np.concatenate([
        img[:s, :s].reshape(-1, 3), img[:s, -s:].reshape(-1, 3),
        img[-s:, :s].reshape(-1, 3), img[-s:, -s:].reshape(-1, 3),
    ]).astype(np.float32)

    corner_mean = corners.mean()
    corner_channel_spread = np.ptp(corners.mean(axis=0))
    saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()

    return 95 <= corner_mean <= 175 and corner_channel_spread < 20 and saturation < 70


def textured_fov_mask(img: np.ndarray) -> np.ndarray:
    """Retina mask for a Ben-Graham image, whose background is flat grey (no texture)."""
    size = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(5, size // 25)
    mean = cv2.blur(gray, (k, k))
    local_std = np.sqrt(np.clip(cv2.blur(gray * gray, (k, k)) - mean * mean, 0, None))

    # The background still carries amplified JPEG noise, so the threshold is
    # relative to the texture found in the centre of the image (always retina).
    c0, c1 = size // 4, 3 * size // 4
    reference = float(np.median(local_std[c0:c1, c0:c1]))
    textured = (local_std > max(6.0, 0.5 * reference)).astype(np.uint8)
    textured = cv2.morphologyEx(textured, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(textured, connectivity=8)
    if n <= 1:
        return np.ones_like(textured, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [hull], -1, 1, thickness=-1)

    # drop the bright ring artefact at the edge of the retina
    e = max(5, int(size * 0.05) | 1)
    filled = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
    return filled.astype(bool)


@dataclass
class PreparedImage:
    """All image variants the pipeline needs."""
    model_input: np.ndarray   # BGR, MODEL_INPUT_SIZE^2, exactly like training data
    display: np.ndarray       # BGR, DISPLAY_SIZE^2, natural-looking image for overlays
    enhanced: np.ndarray      # BGR, DISPLAY_SIZE^2, Ben Graham version (lesion detection)
    fov_mask: np.ndarray      # bool, DISPLAY_SIZE^2, retina area
    already_preprocessed: bool


def prepare_image(img: np.ndarray, model_size: int, display_size: int,
                  already_preprocessed=None) -> PreparedImage:
    if already_preprocessed is None:
        already_preprocessed = looks_preprocessed(img)

    if already_preprocessed:
        square = cv2.resize(img, (display_size, display_size), interpolation=cv2.INTER_AREA)
        model_input = cv2.resize(img, (model_size, model_size), interpolation=cv2.INTER_AREA)
        display = square
        enhanced = square
        fov_mask = textured_fov_mask(square)
    else:
        cropped = crop_black_borders(img)
        model_input = preprocess_for_training(img, model_size)
        display = cv2.resize(cropped, (display_size, display_size), interpolation=cv2.INTER_AREA)
        enhanced = cv2.resize(ben_graham_preprocess(apply_clahe(cropped)),
                              (display_size, display_size), interpolation=cv2.INTER_AREA)
        fov_mask = field_of_view_mask(display)

    return PreparedImage(model_input, display, enhanced, fov_mask, bool(already_preprocessed))
