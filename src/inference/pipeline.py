"""End-to-end analysis of one fundus image.

The model runs ONCE (prediction + Grad-CAM in a single pass); every
explainability module then reuses the same attention map and lesion candidates.
"""

import io
import os

import cv2
import numpy as np
from PIL import Image

from src.config.config import CLASS_NAMES, DISPLAY_SIZE, MODEL_INPUT_SIZE, RESULTS_DIR
from src.explainability.focus_consistency import compute_focus_consistency, generate_focus_consistency
from src.explainability.gradcam import render_heatmap_overlay, upsample_cam
from src.explainability.layered_evidence_map import detect_lesions, generate_layered_evidence, locate_optic_disc
from src.explainability.region_importance_map import compute_region_importance, generate_region_importance
from src.explainability.saliency_evolution import compute_attention_spread, generate_saliency_evolution
from src.inference.predict import get_predictor
from src.preprocessing.resize_normalize import prepare_image
from src.quality_assessment.retinal_quality_overlay import assess_quality, generate_quality_overlay
from src.reporting.encrypt_pdf import encrypt_pdf
from src.reporting.generate_pdf import generate_pdf_report
from src.utils.helpers import new_result_id
from src.visualization.comparison_panel import get_reference_images
from src.visualization.pseudo3d_abstraction import generate_pseudo3d

# file name -> title, in display order
MAPS = {
    "original": "Input Retina",
    "gradcam": "Grad-CAM Attention",
    "evidence": "Layered Evidence Map",
    "consistency": "Focus Consistency",
    "saliency": "Attention Spread",
    "regions": "Region Importance",
    "pseudo3d": "Pseudo-3D Layered View",
    "quality": "Image Quality",
}


class InvalidImage(ValueError):
    pass


MAX_PIXELS = 50_000_000     # refuse decompression bombs before decoding
MAX_SIDE = 3072             # larger photos are downscaled (keeps memory low)


def decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as probe:     # reads only the header
            w, h = probe.size
    except Exception:
        raise InvalidImage("The file is not a readable image.")
    if w * h > MAX_PIXELS:
        raise InvalidImage("Image is too large (maximum 50 megapixels).")

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImage("The file is not a readable image.")
    h, w = img.shape[:2]
    if min(h, w) < 128:
        raise InvalidImage("Image is too small (minimum 128 x 128 pixels).")
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def run_full_analysis(image, already_preprocessed=None, results_dir: str = RESULTS_DIR,
                      predictor=None, include_report: bool = True):
    """image: path, raw bytes or a BGR numpy array. Returns a JSON-serialisable dict."""
    if isinstance(image, (bytes, bytearray)):
        img = decode_image(bytes(image))
    elif isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise InvalidImage(f"Could not read image: {image}")
    else:
        img = image

    result_id = new_result_id()
    out_dir = os.path.join(results_dir, result_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- preprocessing (same as training) ----------------
    prep = prepare_image(img, MODEL_INPUT_SIZE, DISPLAY_SIZE, already_preprocessed)

    # ---------------- model: prediction + Grad-CAM ----------------
    predictor = predictor or get_predictor()
    pred = predictor.predict(prep.model_input)
    cam = upsample_cam(pred["cam"], DISPLAY_SIZE, prep.fov_mask)

    # ---------------- shared evidence ----------------
    optic_disc = locate_optic_disc(prep.enhanced, prep.fov_mask)
    lesions = detect_lesions(prep.enhanced, prep.fov_mask, optic_disc)

    consistency_score, consistency_label = compute_focus_consistency(cam, lesions, prep.fov_mask)
    region = compute_region_importance(cam, prep.fov_mask)
    spread_fraction, attention_label = compute_attention_spread(cam, prep.fov_mask)
    quality, quality_masks = assess_quality(prep.display, prep.fov_mask)

    # ---------------- images ----------------
    images = {
        "original": prep.display,
        "gradcam": render_heatmap_overlay(prep.display, cam),
        "evidence": generate_layered_evidence(prep.display, cam, lesions, optic_disc),
        "consistency": generate_focus_consistency(prep.display, cam, lesions, prep.fov_mask,
                                                  consistency_score, consistency_label),
        "saliency": generate_saliency_evolution(prep.display, cam, lesions, spread_fraction, attention_label),
        "regions": generate_region_importance(prep.display, cam, prep.fov_mask, region),
        "pseudo3d": generate_pseudo3d(prep.display, cam, lesions, prep.fov_mask),
        "quality": generate_quality_overlay(prep.display, prep.fov_mask, quality, quality_masks),
    }
    paths = {}
    for key, image_bgr in images.items():
        path = os.path.join(out_dir, f"{key}.jpg")
        cv2.imwrite(path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths[key] = path

    results = {
        "result_id": result_id,
        "prediction": pred["prediction"],
        "class_index": pred["class_index"],
        "classes": CLASS_NAMES,
        "probabilities": pred["probabilities"],
        "confidence": max(pred["probabilities"]),
        "already_preprocessed": prep.already_preprocessed,
        "consistency_score": consistency_score,
        "consistency_label": consistency_label,
        "attention_spread": spread_fraction,
        "attention_label": attention_label,
        "region_importance": region,
        "quality": quality,
        "lesion_counts": {
            "bright": sum(1 for les in lesions if les["kind"] == "bright"),
            "dark": sum(1 for les in lesions if les["kind"] == "dark"),
        },
        "images": {key: f"/api/results/{result_id}/{key}.jpg" for key in paths},
        "image_titles": MAPS,
        "references": get_reference_images(pred["prediction"]),
    }

    # ---------------- encrypted PDF report ----------------
    if include_report:
        pdf = generate_pdf_report(out_dir, results, {MAPS[k]: paths[k] for k in MAPS})
        _, password = encrypt_pdf(pdf)
        results["report_url"] = f"/api/reports/{result_id}/report.pdf"
        # returned once to the person who ran the analysis; never stored on the server
        results["report_password"] = password
    else:
        results["report_url"] = None
        results["report_password"] = None

    return results
