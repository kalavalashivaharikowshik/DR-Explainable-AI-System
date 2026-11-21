"""Thread-safe, load-once wrapper around the trained classifier.

Two interchangeable backends:

* ``onnx``  (default when models/trained/best_model.onnx exists) - ONNX Runtime,
  no PyTorch needed, ~4x less memory. Grad-CAM is computed exactly from the
  exported feature map and classifier weights (see scripts/export_onnx.py).
* ``torch`` - PyTorch with autograd Grad-CAM (used for development / training
  machines, or when the ONNX file is missing).

Choose explicitly with the environment variable DR_BACKEND=onnx|torch.
"""

import os
import threading

import cv2
import numpy as np

from src.config.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH, ONNX_MODEL_PATH


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
            "Run `git lfs install && git lfs pull` (or scripts/fetch_model.py) to download the model."
        )


def to_input_tensor(model_input_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> normalised float32 1x3xHxW (same as torchvision ToTensor + Normalize)."""
    rgb = cv2.cvtColor(model_input_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _normalise_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0).astype(np.float32)
    peak = float(cam.max())
    return cam / peak if peak > 1e-8 else np.zeros_like(cam)


def _result(probs, class_idx, cam):
    return {
        "prediction": CLASS_NAMES[class_idx],
        "class_index": int(class_idx),
        "probabilities": [float(p) for p in probs],
        "cam": cam,
    }


class OnnxPredictor:
    backend = "onnx"

    def __init__(self, model_path: str = ONNX_MODEL_PATH):
        import onnxruntime as ort

        _check_weights_file(model_path)
        head_path = os.path.splitext(model_path)[0] + "_head.npz"
        if not os.path.exists(head_path):
            raise ModelNotAvailable(f"Classifier weights not found at {head_path}.")

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("DR_THREADS", "1"))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # keep memory low (free hosting tiers have 512 MB): no pre-allocated arenas
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

        head = np.load(head_path)
        self.class_weights = head["weight"]          # num_classes x channels
        self._lock = threading.Lock()

    def predict(self, model_input_bgr: np.ndarray):
        tensor = to_input_tensor(model_input_bgr)
        with self._lock:
            logits, features = self.session.run(["logits", "features"], {"input": tensor})

        probs = _softmax(logits[0].astype(np.float64))
        class_idx = int(np.argmax(probs))
        # Grad-CAM for GAP -> linear heads: weights are the classifier row of the class
        cam = np.tensordot(self.class_weights[class_idx], features[0], axes=(0, 0))
        return _result(probs, class_idx, _normalise_cam(cam))


class TorchPredictor:
    backend = "torch"

    def __init__(self, model_path: str = MODEL_PATH, device: str = None):
        import torch

        from src.explainability.gradcam import GradCAM
        from src.models.classification.efficientnet_model import get_model

        _check_weights_file(model_path)
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = get_model(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        self.gradcam = GradCAM(self.model, self.model.features[-1])
        self._lock = threading.Lock()

    def predict(self, model_input_bgr: np.ndarray):
        tensor = self.torch.from_numpy(to_input_tensor(model_input_bgr)).to(self.device)
        with self._lock:
            probs, class_idx, cam = self.gradcam.generate(tensor)
        return _result(probs, class_idx, cam)


def create_predictor(backend: str = None):
    backend = (backend or os.environ.get("DR_BACKEND", "")).lower()
    if not backend:
        backend = "onnx" if os.path.exists(ONNX_MODEL_PATH) else "torch"
    if backend == "onnx":
        return OnnxPredictor()
    if backend == "torch":
        return TorchPredictor()
    raise ValueError("DR_BACKEND must be 'onnx' or 'torch'")


# backwards-compatible name
DRPredictor = TorchPredictor

_predictor = None
_predictor_lock = threading.Lock()


def get_predictor():
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = create_predictor()
    return _predictor
