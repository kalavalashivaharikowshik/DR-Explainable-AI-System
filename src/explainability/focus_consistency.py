"""Focus consistency: does the model's strongest attention fall on lesion candidates?

score = share of the high-attention area (CAM >= 0.5) that lies within the
neighbourhood of a detected lesion candidate.
"""

import cv2
import numpy as np

HIGH_ATTENTION = 0.5


def build_lesion_mask(shape, lesions, margin: int):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for les in lesions:
        cv2.circle(mask, (les["x"], les["y"]), int(les["r"] * 1.5) + margin, 1, -1)
    return mask.astype(bool)


def compute_focus_consistency(cam: np.ndarray, lesions, fov_mask: np.ndarray):
    """Return (score in [0,1] or None, label)."""
    if not lesions:
        return None, "No lesion candidates"

    margin = int(cam.shape[0] * 0.02)
    lesion_mask = build_lesion_mask(cam.shape, lesions, margin)
    attention = (cam >= HIGH_ATTENTION) & fov_mask

    attention_area = int(attention.sum())
    if attention_area == 0:
        return 0.0, "Low Agreement"

    score = float((attention & lesion_mask).sum()) / attention_area

    if score > 0.4:
        label = "High Agreement"
    elif score > 0.15:
        label = "Partial Agreement"
    else:
        label = "Low Agreement"

    return score, label


def generate_focus_consistency(display, cam, lesions, fov_mask, score, label):
    """Image: attention contour + lesion neighbourhoods, coloured by agreement."""
    out = display.copy()
    size = cam.shape[0]
    margin = int(size * 0.02)

    lesion_mask = build_lesion_mask(cam.shape, lesions, margin) if lesions else np.zeros(cam.shape, bool)
    attention = (cam >= HIGH_ATTENTION) & fov_mask
    overlap = attention & lesion_mask

    tint = out.copy()
    tint[attention & ~lesion_mask] = (255, 140, 0)      # attention only -> blue
    tint[lesion_mask & ~attention] = (0, 200, 255)       # lesion only   -> amber
    tint[overlap] = (0, 220, 0)                          # agreement     -> green
    out = cv2.addWeighted(out, 0.55, tint, 0.45, 0)

    contours, _ = cv2.findContours(attention.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    color = {"High Agreement": (0, 220, 0), "Partial Agreement": (0, 220, 255)}.get(label, (80, 80, 255))
    text = label if score is None else f"{label} ({score * 100:.0f}%)"
    _label(out, text, color)
    return out


def _label(img, text, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
