"""Thread-safe, load-once wrapper around the trained classifier."""

import os
import threading

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.config.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH
from src.explainability.gradcam import GradCAM
from src.models.classification.efficientnet_model import get_model


class ModelNotAvailable(RuntimeError):
    pass


def _check_weights_file(path: str) -> None:
    if not os.path.exists(path):
        raise ModelNotAvailable(f"Model weights not found at {path}.")
    with open(path, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise ModelNotAvailable(
            f"{path} is a Git LFS pointer, not the real weights. "
            "Run `git lfs install && git lfs pull` to download the model."
        )


class DRPredictor:

    def __init__(self, model_path: str = MODEL_PATH, device: str = None):
        _check_weights_file(model_path)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = get_model(pretrained=False)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # last convolutional block of EfficientNet
        self.gradcam = GradCAM(self.model, self.model.features[-1])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # GradCAM keeps per-call state in hooks -> serialise access
        self._lock = threading.Lock()

    def predict(self, model_input_bgr: np.ndarray):
        """model_input_bgr: preprocessed image at MODEL_INPUT_SIZE (see resize_normalize).

        Returns dict(prediction, class_index, probabilities, cam).
        """
        rgb = cv2.cvtColor(model_input_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with self._lock:
            probs, class_idx, cam = self.gradcam.generate(tensor)

        return {
            "prediction": CLASS_NAMES[class_idx],
            "class_index": class_idx,
            "probabilities": [float(p) for p in probs],
            "cam": cam,
        }


_predictor = None
_predictor_lock = threading.Lock()


def get_predictor() -> DRPredictor:
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = DRPredictor()
    return _predictor
