import os
import cv2
import numpy as np

from src.explainability.gradcam import run_gradcam
from src.explainability.layered_evidence_map import detect_lesions

OUTPUT_DIR = "outputs/focus_consistency"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_lesion_mask(image, lesions):

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for x, y, r in lesions:
        cv2.circle(mask, (x, y), int(r*1.5), 255, -1)

    return mask


def build_attention_mask(heatmap_img):

    gray = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)

    # normalize
    gray = gray / 255.0

    # threshold top attention
    mask = (gray > 0.6).astype(np.uint8) * 255

    return mask


def compute_overlap(attention_mask, lesion_mask):

    intersection = cv2.bitwise_and(attention_mask, lesion_mask)

    union = cv2.bitwise_or(attention_mask, lesion_mask)

    inter_area = np.sum(intersection > 0)
    union_area = np.sum(union > 0)

    if union_area == 0:
        return 0

    return inter_area / union_area


def generate_focus_consistency(image_path):

    gradcam_result = run_gradcam(image_path)

    prediction = gradcam_result["prediction"]
    heatmap_path = gradcam_result["heatmap_path"]

    original = cv2.imread(image_path)
    heatmap_img = cv2.imread(heatmap_path)

    lesions = detect_lesions(original)

    lesion_mask = build_lesion_mask(original, lesions)
    attention_mask = build_attention_mask(heatmap_img)

    score = compute_overlap(attention_mask, lesion_mask)

    output = heatmap_img.copy()

    if score > 0.4:
        label = "High Agreement"
        color = (0,255,0)

    elif score > 0.15:
        label = "Partial Agreement"
        color = (0,255,255)

    else:
        label = "Low Agreement"
        color = (0,0,255)

    cv2.putText(
        output,
        f"Consistency: {label}",
        (30,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"consistency_{filename}"
    )

    cv2.imwrite(output_path, output)

    return {
        "prediction": prediction,
        "consistency_score": score,
        "consistency_label": label,
        "output_path": output_path
    }