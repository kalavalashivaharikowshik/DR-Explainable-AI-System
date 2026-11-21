"""Pseudo-3D layered retina view.

Layers (bottom -> top):
  1. dimmed retina
  2. blood vessels (morphological black-hat on the green channel)
  3. lesion candidates
  4. Grad-CAM attention rendered as a shaded relief ("height" = attention),
     which gives the attention map a 3D look without a 3D renderer.
"""

import cv2
import numpy as np


def extract_vessels(image_bgr: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    size = image_bgr.shape[0]
    green = image_bgr[..., 1]
    green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)

    k = max(9, int(size / 30) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
    blackhat[~fov_mask] = 0

    values = blackhat[fov_mask]
    thresh = np.percentile(values, 88) if values.size else 255
    vessels = (blackhat > max(thresh, 8)).astype(np.uint8)

    # remove specks: vessels are connected, elongated structures
    n, labels, stats, _ = cv2.connectedComponentsWithStats(vessels, connectivity=8)
    min_area = int(40 * (size / 512) ** 2)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def shaded_relief(cam: np.ndarray, strength: float = 6.0) -> np.ndarray:
    """Hill-shading of the attention surface, light from the top-left. Returns [0,1]."""
    smooth = cv2.GaussianBlur(cam, (0, 0), 3)
    gy, gx = np.gradient(smooth * cam.shape[0] / 64.0)
    shade = 0.6 + strength * (-gx - gy) / 2
    return np.clip(shade, 0.25, 1.4)


def generate_pseudo3d(display, cam, lesions, fov_mask):
    base = (display.astype(np.float32) * 0.45)

    vessels = extract_vessels(display, fov_mask)
    base[vessels] = (230, 230, 230)

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET).astype(np.float32)
    relief = shaded_relief(cam)[..., None]
    alpha = (np.clip(cam, 0, 1) ** 0.8 * 0.75)[..., None]
    out = base * (1 - alpha) + heat * relief * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    for les in lesions:
        color = (0, 255, 255) if les["kind"] == "bright" else (60, 60, 255)
        cv2.circle(out, (les["x"], les["y"]), les["r"], color, 1, cv2.LINE_AA)

    out[~fov_mask] = (15, 15, 15)
    return out
