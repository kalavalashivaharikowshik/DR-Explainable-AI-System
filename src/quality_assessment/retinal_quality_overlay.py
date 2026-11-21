"""Image quality assessment for fundus photographs.

* Low-detail / blurred areas: 32 px blocks whose local Laplacian variance is
  far below the image's typical block (relative, so it works on raw and
  pre-processed images alike)        -> grey
* Over-exposed / glare (very bright) -> yellow
* Under-exposed (very dark)          -> blue

A 0-100 quality score and Good / Fair / Poor label summarise the result.
"""

import cv2
import numpy as np

BLOCK = 32


def detect_low_detail_blocks(gray: np.ndarray, fov_mask: np.ndarray, block: int = BLOCK):
    h, w = gray.shape
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mask = np.zeros_like(fov_mask)
    variances, cells = [], []

    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            inside = fov_mask[y:y + block, x:x + block]
            if inside.mean() < 0.8:          # skip blocks on the retina edge
                continue
            variances.append(lap[y:y + block, x:x + block].var())
            cells.append((y, x))

    if not variances:
        return mask, 0.0

    reference = np.median(variances)
    flagged = 0
    for v, (y, x) in zip(variances, cells):
        if v < 0.2 * reference:
            mask[y:y + block, x:x + block] = True
            flagged += 1

    return mask & fov_mask, flagged / len(variances)


def assess_quality(image_bgr: np.ndarray, fov_mask: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    value = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[..., 2]

    blur_mask, blur_fraction = detect_low_detail_blocks(gray, fov_mask)
    over_mask = (value > 245) & fov_mask
    under_mask = (value < 25) & fov_mask

    fov_area = max(int(fov_mask.sum()), 1)
    over_fraction = over_mask.sum() / fov_area
    under_fraction = under_mask.sum() / fov_area
    fov_coverage = fov_area / fov_mask.size

    penalty = 2.5 * over_fraction + 1.2 * under_fraction + 1.0 * blur_fraction
    if fov_coverage < 0.45:          # retina only partly visible
        penalty += 0.3
    score = float(np.clip(100 * (1 - penalty), 0, 100))
    label = "Good" if score >= 75 else "Fair" if score >= 50 else "Poor"

    return {
        "score": round(score, 1),
        "label": label,
        "blur_fraction": float(blur_fraction),
        "overexposed_fraction": float(over_fraction),
        "underexposed_fraction": float(under_fraction),
    }, (blur_mask, over_mask, under_mask)


def generate_quality_overlay(image_bgr, fov_mask, quality, masks):
    blur_mask, over_mask, under_mask = masks
    tint = image_bgr.copy()
    tint[blur_mask] = (128, 128, 128)
    tint[under_mask] = (200, 80, 0)
    tint[over_mask] = (0, 255, 255)
    out = cv2.addWeighted(image_bgr, 0.45, tint, 0.55, 0)

    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (20, 20, 20), -1)
    color = {"Good": (0, 220, 0), "Fair": (0, 220, 255)}.get(quality["label"], (80, 80, 255))
    cv2.putText(out, f"Image quality: {quality['label']} ({quality['score']:.0f}/100)", (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out
