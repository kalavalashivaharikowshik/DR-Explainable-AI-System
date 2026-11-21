import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def synthetic_fundus(size=600, preprocessed=False, seed=0):
    """A fake retina: orange disc on black, bright optic disc, a few vessels and spots."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size + 100, 3), np.uint8)
    c = (img.shape[1] // 2, size // 2)
    cv2.circle(img, c, int(size * 0.45), (40, 90, 200), -1)
    noise = rng.normal(0, 6, img.shape)
    img = np.clip(img + noise * (img > 0), 0, 255).astype(np.uint8)
    cv2.circle(img, (c[0] + int(size * 0.25), c[1]), int(size * 0.05), (180, 220, 250), -1)  # optic disc
    for k in range(5):
        cv2.line(img, (c[0] + int(size * 0.25), c[1]), (c[0] - 200 + 80 * k, c[1] - 220 + 100 * k), (20, 40, 120), 4)
    for x, y in [(c[0] - 80, c[1] + 40), (c[0] - 40, c[1] - 90), (c[0] + 30, c[1] + 120)]:
        cv2.circle(img, (x, y), 7, (150, 230, 255), -1)       # bright spots
    cv2.circle(img, (c[0] - 120, c[1] - 20), 6, (10, 10, 60), -1)  # dark spot
    if preprocessed:
        from src.preprocessing.resize_normalize import preprocess_for_training
        img = preprocess_for_training(img, 300)
    return img


@pytest.fixture
def raw_image():
    return synthetic_fundus()


@pytest.fixture
def processed_image():
    return synthetic_fundus(preprocessed=True)


class FakePredictor:
    """Stands in for the real model so tests run without the 43 MB weights."""

    def predict(self, model_input):
        h, w = model_input.shape[:2]
        cam = np.zeros((10, 10), np.float32)
        cam[3:6, 2:5] = 1.0
        cam[6, 6] = 0.4
        return {"prediction": "Moderate", "class_index": 2,
                "probabilities": [0.05, 0.1, 0.7, 0.1, 0.05], "cam": cam}


@pytest.fixture
def fake_predictor():
    return FakePredictor()


def model_weights_available():
    path = os.path.join(ROOT, "models", "trained", "best_model.pth")
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        return not f.read(64).startswith(b"version https://git-lfs")
