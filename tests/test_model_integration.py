"""Runs the real trained model. Skipped when the weights are not downloaded (git lfs pull)."""

import importlib.util

import numpy as np
import pytest

from tests.conftest import model_weights_available, synthetic_fundus

has_onnx = model_weights_available("best_model.onnx") and importlib.util.find_spec("onnxruntime")
has_torch = model_weights_available("best_model.pth") and importlib.util.find_spec("torch")


@pytest.mark.skipif(not has_onnx, reason="ONNX model or onnxruntime not available")
def test_full_pipeline_with_onnx_backend(tmp_path):
    from src.inference.pipeline import run_full_analysis
    from src.inference.predict import OnnxPredictor

    result = run_full_analysis(synthetic_fundus(), results_dir=str(tmp_path), predictor=OnnxPredictor())
    assert result["prediction"] in result["classes"]
    assert abs(sum(result["probabilities"]) - 1) < 1e-4


@pytest.mark.skipif(not (has_onnx and has_torch), reason="needs both PyTorch and ONNX models")
def test_onnx_matches_pytorch_prediction_and_gradcam():
    from src.inference.predict import OnnxPredictor, TorchPredictor
    from src.preprocessing.resize_normalize import prepare_image

    onnx_p, torch_p = OnnxPredictor(), TorchPredictor(device="cpu")
    for seed in range(3):
        x = prepare_image(synthetic_fundus(seed=seed), 300, 512).model_input
        a, b = onnx_p.predict(x), torch_p.predict(x)
        assert a["class_index"] == b["class_index"]
        assert np.allclose(a["probabilities"], b["probabilities"], atol=1e-4)
        assert np.allclose(a["cam"], b["cam"], atol=1e-3)


@pytest.mark.skipif(not importlib.util.find_spec("torch"), reason="PyTorch not installed")
def test_numpy_transform_matches_torchvision():
    import cv2
    from torchvision import transforms

    from src.config.config import IMAGENET_MEAN, IMAGENET_STD
    from src.inference.predict import to_input_tensor

    img = synthetic_fundus(preprocessed=True)
    ref = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).numpy()
    assert np.allclose(to_input_tensor(img), ref, atol=1e-5)
