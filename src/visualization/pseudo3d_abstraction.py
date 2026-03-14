import os
import cv2
import numpy as np

from src.explainability.gradcam import run_gradcam
from src.explainability.layered_evidence_map import detect_lesions

OUTPUT_DIR = "outputs/pseudo3d"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_vessels(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    vessels = cv2.Canny(blur,40,120)

    vessels = cv2.dilate(vessels,np.ones((2,2)))

    return vessels


def generate_pseudo3d(image_path):

    gradcam = run_gradcam(image_path)

    heatmap_path = gradcam["heatmap_path"]

    original = cv2.imread(image_path)

    heatmap = cv2.imread(heatmap_path)

    vessels = extract_vessels(original)

    lesions = detect_lesions(original)

    h,w,_ = original.shape

    canvas = np.zeros((h,w,3),dtype=np.uint8)

    # layer 1 base retina
    base = cv2.addWeighted(original,0.6,canvas,0.4,0)

    # layer 2 vessels
    vessel_layer = base.copy()
    vessel_layer[vessels>0] = (255,255,255)

    # layer 3 lesions
    lesion_layer = vessel_layer.copy()

    for x,y,r in lesions:

        cv2.circle(lesion_layer,(x,y),r,(0,255,255),2)

    # layer 4 attention
    final = cv2.addWeighted(lesion_layer,0.7,heatmap,0.3,0)

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"pseudo3d_{filename}"
    )

    cv2.imwrite(output_path,final)

    return {
        "pseudo3d_map": output_path
    }