import os
import cv2
import numpy as np

from src.explainability.gradcam import run_gradcam

OUTPUT_DIR = "outputs/evidence_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Lesion Candidate Detection
# -----------------------------
def detect_lesions(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(3.0,(8,8))
    enhanced = clahe.apply(gray)

    _,thresh = cv2.threshold(enhanced,220,255,cv2.THRESH_BINARY)

    thresh = cv2.medianBlur(thresh,5)

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lesions = []

    for c in contours:

        if cv2.contourArea(c) > 20:

            (x,y),radius = cv2.minEnclosingCircle(c)

            lesions.append((int(x),int(y),int(radius)))

    return lesions


# -----------------------------
# Retina Region Overlay
# -----------------------------
def draw_retina_regions(image):

    h,w,_ = image.shape

    center = (w//2 , h//2)

    overlay = image.copy()

    # macula region
    cv2.circle(overlay,center,80,(255,0,0),2)

    # mid retina
    cv2.circle(overlay,center,160,(0,255,0),2)

    # peripheral retina
    cv2.circle(overlay,center,240,(0,255,255),2)

    return overlay


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def generate_layered_evidence(image_path):

    gradcam_result = run_gradcam(image_path)

    prediction = gradcam_result["prediction"]

    heatmap_path = gradcam_result["heatmap_path"]

    original = cv2.imread(image_path)

    gradcam_img = cv2.imread(heatmap_path)

    # draw retina zones
    region_img = draw_retina_regions(gradcam_img)

    # detect lesion points
    lesions = detect_lesions(original)

    for x,y,r in lesions:

        cv2.circle(region_img,(x,y),r,(0,255,255),2)

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"evidence_{filename}"
    )

    cv2.imwrite(output_path,region_img)

    return {
        "prediction": prediction,
        "evidence_map": output_path
    }