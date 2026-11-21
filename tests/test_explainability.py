import cv2
import numpy as np
import torch

from src.explainability.focus_consistency import compute_focus_consistency
from src.explainability.gradcam import GradCAM, upsample_cam
from src.explainability.layered_evidence_map import detect_lesions, locate_optic_disc
from src.explainability.region_importance_map import compute_region_importance
from src.explainability.saliency_evolution import compute_attention_spread
from src.preprocessing.resize_normalize import prepare_image
from src.quality_assessment.retinal_quality_overlay import assess_quality


def disc_mask(size=200, r=90):
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), r, 1, -1)
    return m.astype(bool)


def test_region_importance_sums_to_one_and_finds_centre():
    fov = disc_mask()
    cam = np.zeros((200, 200), np.float32)
    cam[95:105, 95:105] = 1.0
    r = compute_region_importance(cam, fov)
    assert abs(sum(r.values()) - 1) < 1e-6
    assert r["central"] > 0.99


def test_region_importance_empty_cam():
    assert compute_region_importance(np.zeros((200, 200), np.float32), disc_mask()) == \
        {"central": 0.0, "mid": 0.0, "peripheral": 0.0}


def test_attention_spread_labels():
    fov = disc_mask()
    focused = np.zeros((200, 200), np.float32)
    focused[100:106, 100:106] = 1
    assert compute_attention_spread(focused, fov)[1] == "Focused Attention"
    diffuse = np.ones((200, 200), np.float32)
    assert compute_attention_spread(diffuse, fov)[1] == "Diffuse Attention"
    assert compute_attention_spread(np.zeros((200, 200), np.float32), fov)[1] == "No Attention"


def test_focus_consistency_agreement():
    fov = disc_mask()
    cam = np.zeros((200, 200), np.float32)
    cam[60:80, 60:80] = 1
    on = [{"x": 70, "y": 70, "r": 10, "kind": "bright"}]
    off = [{"x": 150, "y": 150, "r": 4, "kind": "bright"}]
    assert compute_focus_consistency(cam, on, fov)[1] == "High Agreement"
    assert compute_focus_consistency(cam, off, fov)[1] == "Low Agreement"
    assert compute_focus_consistency(cam, [], fov) == (None, "No lesion candidates")


def test_upsample_cam_is_normalised_and_masked():
    cam = np.random.rand(10, 10).astype(np.float32)
    fov = disc_mask(64, 28)
    big = upsample_cam(cam, 64, fov)
    assert big.shape == (64, 64)
    assert abs(big.max() - 1) < 1e-6 and big.min() >= 0
    assert big[0, 0] == 0


def test_upsample_zero_cam_gives_zeros_not_nan():
    big = upsample_cam(np.zeros((10, 10), np.float32), 32)
    assert not np.isnan(big).any() and big.max() == 0


def test_gradcam_on_tiny_cnn():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 5))
    model.eval()
    probs, idx, cam = GradCAM(model, model[1]).generate(torch.rand(1, 3, 16, 16))
    assert len(probs) == 5 and abs(probs.sum() - 1) < 1e-5
    assert 0 <= idx < 5
    assert cam.shape == (16, 16) and cam.min() >= 0 and cam.max() <= 1
    assert not np.isnan(cam).any()


def test_lesion_detection_and_quality_run_on_synthetic(raw_image):
    prep = prepare_image(raw_image, 300, 512)
    disc = locate_optic_disc(prep.enhanced, prep.fov_mask)
    lesions = detect_lesions(prep.enhanced, prep.fov_mask, disc)
    assert all(les["kind"] in ("bright", "dark") for les in lesions)
    assert len(lesions) <= 50
    quality, masks = assess_quality(prep.display, prep.fov_mask)
    assert 0 <= quality["score"] <= 100 and quality["label"] in ("Good", "Fair", "Poor")
