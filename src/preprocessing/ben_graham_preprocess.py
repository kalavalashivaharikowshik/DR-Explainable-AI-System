import cv2
import numpy as np


def ben_graham_preprocess(img: np.ndarray, sigma: float = 10) -> np.ndarray:
    """Ben Graham's local-average colour subtraction (Kaggle DR 2015 winner).

    Subtracts a Gaussian-blurred copy so lesions and vessels stand out on a
    neutral grey (~128) background.
    """
    return cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), sigma),
        -4, 128,
    )
