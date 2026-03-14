import os
import cv2
import numpy as np

from src.explainability.gradcam import run_gradcam
from src.explainability.layered_evidence_map import detect_lesions

OUTPUT_DIR = "outputs/saliency_evolution"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_attention_behavior(heatmap, lesions):

    spreads = []

    h, w = heatmap.shape

    for x, y, r in lesions:

        center_val = heatmap[y, x]

        spread = 0

        for radius in range(5, 80, 5):

            points = []

            for angle in range(0, 360, 30):

                px = int(x + radius * np.cos(np.radians(angle)))
                py = int(y + radius * np.sin(np.radians(angle)))

                if 0 <= px < w and 0 <= py < h:
                    points.append(heatmap[py, px])

            if len(points) > 0 and np.mean(points) > center_val * 0.5:
                spread = radius

        spreads.append(spread)

    if len(spreads) == 0:
        return "Diffuse Attention"

    avg_spread = np.mean(spreads)

    if avg_spread < 25:
        return "Focused Attention"
    elif avg_spread < 50:
        return "Moderate Spread"
    else:
        return "Diffuse Attention"


def draw_halo(image, x, y):

    overlay = image.copy()

    # create soft glow
    cv2.circle(overlay, (x, y), 60, (0,255,255), -1)

    overlay = cv2.GaussianBlur(overlay, (51,51), 0)

    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return result


def generate_saliency_evolution(image_path):

    gradcam_result = run_gradcam(image_path)

    heatmap_path = gradcam_result["heatmap_path"]
    prediction = gradcam_result["prediction"]

    original = cv2.imread(image_path)
    heatmap_img = cv2.imread(heatmap_path)

    gray = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)

    lesions = detect_lesions(original)

    output = heatmap_img.copy()

    for x, y, r in lesions:

        output = draw_halo(output, x, y)

        cv2.circle(output, (x, y), 4, (255,255,255), -1)

    behavior = compute_attention_behavior(gray, lesions)

    # label color
    if behavior == "Focused Attention":
        color = (0,255,0)
    elif behavior == "Moderate Spread":
        color = (0,255,255)
    else:
        color = (0,0,255)

    text = f"Saliency: {behavior}"

    cv2.putText(
        output,
        text,
        (20,30),                 # safer position
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,                     # smaller font
        color,
        2,
        cv2.LINE_AA
    )

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"saliency_{filename}"
    )

    cv2.imwrite(output_path, output)

    return {
        "prediction": prediction,
        "behavior": behavior,
        "output_path": output_path
    }