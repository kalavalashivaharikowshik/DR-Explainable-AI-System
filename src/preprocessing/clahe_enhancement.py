import cv2
import numpy as np


def apply_clahe(img: np.ndarray, clip_limit: float = 3.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the L channel (LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
