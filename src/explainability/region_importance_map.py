"""Share of Grad-CAM attention falling in central / mid / peripheral retina.

Zones are concentric rings scaled to the retina's radius (assumes a
macula-centred photograph, the standard screening field):
    central     0   - 0.25 R   (macula / posterior pole)
    mid         0.25 - 0.6 R
    peripheral  0.6 - 1.0 R
"""

import cv2
import numpy as np

ZONES = (("central", 0.25), ("mid", 0.60), ("peripheral", 1.0))
ZONE_COLORS = {"central": (255, 120, 0), "mid": (0, 220, 0), "peripheral": (0, 220, 255)}


def fov_geometry(fov_mask: np.ndarray):
    ys, xs = np.nonzero(fov_mask)
    if len(xs) == 0:
        h, w = fov_mask.shape
        return (w // 2, h // 2), min(h, w) // 2
    cx, cy = xs.mean(), ys.mean()
    radius = np.sqrt(len(xs) / np.pi)          # radius of a disc with the same area
    return (int(cx), int(cy)), int(radius)


def compute_region_importance(cam: np.ndarray, fov_mask: np.ndarray):
    """Vectorised. Returns dict zone -> fraction (sums to 1, or all 0 if no attention)."""
    (cx, cy), radius = fov_geometry(fov_mask)
    h, w = cam.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1)

    weights = cam * fov_mask
    total = float(weights.sum())
    result = {}
    inner = 0.0
    for name, outer in ZONES:
        ring = (dist >= inner) & (dist < outer) if name != "peripheral" else (dist >= inner)
        result[name] = float(weights[ring].sum()) / total if total > 0 else 0.0
        inner = outer
    return result


def generate_region_importance(display, cam, fov_mask, importance):
    out = display.copy()
    (cx, cy), radius = fov_geometry(fov_mask)

    for name, outer in ZONES[:-1]:
        cv2.circle(out, (cx, cy), int(outer * radius), ZONE_COLORS[name], 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), radius, ZONE_COLORS["peripheral"], 2, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (215, 82), (20, 20, 20), -1)
    for i, (name, _) in enumerate(ZONES):
        cv2.putText(out, f"{name.capitalize()}: {importance[name] * 100:.0f}%", (10, 22 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_COLORS[name], 1, cv2.LINE_AA)
    return out
