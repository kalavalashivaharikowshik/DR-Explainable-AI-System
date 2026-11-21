"""Attention spread: how concentrated is the model's attention?

We find the smallest share of the retina that contains half of the total
Grad-CAM attention mass:
    < 10 % of the retina  -> Focused Attention
    < 25 %                -> Moderate Spread
    otherwise             -> Diffuse Attention

The image shows attention iso-contours at 25 / 50 / 75 % of the peak, i.e. how
attention "evolves" from the broad context to the most decisive spots.
"""

import cv2
import numpy as np

LEVELS = ((0.25, (255, 160, 0)), (0.50, (0, 220, 255)), (0.75, (60, 60, 255)))


def compute_attention_spread(cam: np.ndarray, fov_mask: np.ndarray):
    values = cam[fov_mask]
    total = values.sum()
    if total <= 0:
        return 1.0, "No Attention"

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    n_half = int(np.searchsorted(cumulative, 0.5 * total)) + 1
    fraction = n_half / len(values)

    if fraction < 0.10:
        label = "Focused Attention"
    elif fraction < 0.25:
        label = "Moderate Spread"
    else:
        label = "Diffuse Attention"
    return float(fraction), label


def generate_saliency_evolution(display, cam, lesions, fraction, label):
    out = (display * 0.8).astype(np.uint8)

    for level, color in LEVELS:
        mask = (cam >= level).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)

    for les in lesions:
        cv2.circle(out, (les["x"], les["y"]), 2, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (20, 20, 20), -1)
    color = {"Focused Attention": (0, 220, 0), "Moderate Spread": (0, 220, 255)}.get(label, (80, 80, 255))
    cv2.putText(out, f"{label}: 50% of attention in {fraction * 100:.0f}% of retina", (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
