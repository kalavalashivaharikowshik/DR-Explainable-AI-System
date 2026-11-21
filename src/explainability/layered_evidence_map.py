"""Lesion *candidate* detection + layered evidence map.

This is a classical image-processing heuristic, not a trained lesion
segmentation model. It works on the Ben-Graham enhanced image, where the
background is neutral grey (~128):

* bright candidates  - pixels much brighter than background (exudates /
  cotton-wool spots, but also reflections)
* dark candidates    - compact, roundish spots much darker than background
  (haemorrhages / microaneurysms); long thin dark structures (vessels) are
  rejected by a shape test
* the optic disc (brightest large region) is located and excluded

Results are labelled "candidates" everywhere in the UI and report.
"""

import cv2
import numpy as np

from src.explainability.gradcam import render_heatmap_overlay
from src.visualization.pseudo3d_abstraction import extract_vessels

BRIGHT_COLOR = (0, 255, 255)   # yellow (BGR)
DARK_COLOR = (60, 60, 255)     # red
DISC_COLOR = (255, 255, 255)   # white


def locate_optic_disc(enhanced: np.ndarray, fov_mask: np.ndarray):
    """Return (x, y, radius) of the optic disc estimate."""
    size = enhanced.shape[0]
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), size / 40)
    blurred[~fov_mask] = 0
    _, _, _, (x, y) = cv2.minMaxLoc(blurred)
    return int(x), int(y), int(size * 0.085)


def _components(mask, min_area, max_area):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            yield i, labels, stats[i], centroids[i]


MAX_PER_KIND = 25


def detect_lesions(enhanced: np.ndarray, fov_mask: np.ndarray, optic_disc=None):
    """Return a list of dicts: {x, y, r, kind: 'bright' | 'dark', contrast}.

    Thresholds adapt to each image: a candidate must stand out from its local
    background by several times the image's typical texture amplitude (MAD).
    At most MAX_PER_KIND of the strongest candidates of each kind are kept.
    """
    size = enhanced.shape[0]
    area_scale = (size / 512.0) ** 2

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), size / 200)
    background = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), max(31, (size // 8) | 1)).astype(np.float32)
    diff = smooth - background

    if optic_disc is None:
        optic_disc = locate_optic_disc(enhanced, fov_mask)
    dx, dy, dr = optic_disc

    yy, xx = np.mgrid[:size, :size]
    # ignore a band along the retina edge (illumination fall-off / filter ring artefacts)
    inner = cv2.erode(fov_mask.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(size * 0.06) | 1, int(size * 0.06) | 1))).astype(bool)
    valid = inner & (((xx - dx) ** 2 + (yy - dy) ** 2) > (2.0 * dr) ** 2)

    vessels = cv2.dilate(extract_vessels(enhanced, fov_mask).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    values = diff[valid]
    if values.size == 0:
        return []
    mad = float(np.median(np.abs(values - np.median(values)))) + 1e-6

    lesions = []
    for kind, mask, min_area, max_area in (
        ("bright", (diff > max(26.0, 6.0 * mad)) & valid, 12, 6000),
        ("dark", (diff < -max(24.0, 5.5 * mad)) & valid, 10, 2500),
    ):
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        found = []
        for i, labels, st, (cx, cy) in _components(mask, min_area * area_scale, max_area * area_scale):
            w, h, area = st[cv2.CC_STAT_WIDTH], st[cv2.CC_STAT_HEIGHT], st[cv2.CC_STAT_AREA]
            elongation = max(w, h) / max(1, min(w, h))
            fill = area / float(w * h)
            if elongation > 3.0 or (kind == "dark" and (elongation > 2.2 or fill < 0.45)):
                continue  # thin / branching structures are vessels or reflexes, not lesions
            if kind == "dark" and vessels[labels == i].mean() > 0.4:
                continue  # sits on a detected vessel
            contrast = float(np.abs(diff[labels == i]).mean())
            r = max(4, int(0.5 * max(w, h)) + 3)
            found.append({"x": int(cx), "y": int(cy), "r": r, "kind": kind,
                          "contrast": round(contrast, 1), "_rank": contrast * np.sqrt(area)})
        found.sort(key=lambda d: d["_rank"], reverse=True)
        for d in found[:MAX_PER_KIND]:
            d.pop("_rank")
            lesions.append(d)

    return lesions


def generate_layered_evidence(display: np.ndarray, cam: np.ndarray, lesions, optic_disc):
    """GradCAM attention + lesion candidates + optic disc marker."""
    out = render_heatmap_overlay(display, cam, alpha=0.35)

    dx, dy, dr = optic_disc
    cv2.circle(out, (dx, dy), dr, DISC_COLOR, 1, cv2.LINE_AA)
    cv2.putText(out, "OD", (dx - 10, dy - dr - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, DISC_COLOR, 1, cv2.LINE_AA)

    for les in lesions:
        color = BRIGHT_COLOR if les["kind"] == "bright" else DARK_COLOR
        cv2.circle(out, (les["x"], les["y"]), les["r"], color, 2, cv2.LINE_AA)

    return out
