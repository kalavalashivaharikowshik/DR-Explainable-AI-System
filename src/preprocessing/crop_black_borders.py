import cv2
import numpy as np


def crop_black_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop the dark area around the circular fundus field of view."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > threshold))

    if coords.shape[0] == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # +1 so the last row / column is kept
    return img[y0:y1 + 1, x0:x1 + 1]


def field_of_view_mask(img: np.ndarray, threshold: int = 10, erode_frac: float = 0.02) -> np.ndarray:
    """Boolean mask of the circular retina area (excludes the black background).

    The mask is eroded slightly so the bright rim that Ben Graham filtering
    creates at the edge of the circle is not mistaken for retina.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8)

    # keep only the largest connected region (the retina)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)

    # fill holes (dark lesions near the edge must stay inside the mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)

    k = max(3, int(min(img.shape[:2]) * erode_frac) | 1)
    filled = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    return filled.astype(bool)
