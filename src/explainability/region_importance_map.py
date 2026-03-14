import os
import cv2
import numpy as np

from src.explainability.gradcam import run_gradcam

OUTPUT_DIR = "outputs/region_importance"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_region_importance(heatmap):

    h, w = heatmap.shape

    center = (w // 2, h // 2)

    macula_r = 80
    mid_r = 160

    macula_score = 0
    mid_score = 0
    peripheral_score = 0

    total = 0

    for y in range(h):
        for x in range(w):

            value = heatmap[y, x]

            if value <= 0:
                continue

            dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)

            if dist <= macula_r:
                macula_score += value

            elif dist <= mid_r:
                mid_score += value

            else:
                peripheral_score += value

            total += value

    if total == 0:
        return 0,0,0

    macula = macula_score / total
    mid = mid_score / total
    peripheral = peripheral_score / total

    return macula, mid, peripheral


def generate_region_importance(image_path):

    gradcam_result = run_gradcam(image_path)

    heatmap_path = gradcam_result["heatmap_path"]
    prediction = gradcam_result["prediction"]

    original = cv2.imread(image_path)

    heatmap_img = cv2.imread(heatmap_path)

    gray = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY) / 255.0

    macula, mid, peripheral = compute_region_importance(gray)

    h, w, _ = original.shape
    center = (w//2, h//2)

    output = original.copy()

    cv2.circle(output, center, 80, (255,0,0), 2)
    cv2.circle(output, center, 160, (0,255,0), 2)

    cv2.putText(output, f"Macula: {macula:.2f}",
                (20,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,0,0),2)

    cv2.putText(output, f"Mid Retina: {mid:.2f}",
                (20,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,0),2)

    cv2.putText(output, f"Peripheral: {peripheral:.2f}",
                (20,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,255),2)

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"regions_{filename}"
    )

    cv2.imwrite(output_path, output)

    return {
        "prediction": prediction,
        "macula_importance": macula,
        "mid_importance": mid,
        "peripheral_importance": peripheral,
        "output_path": output_path
    }