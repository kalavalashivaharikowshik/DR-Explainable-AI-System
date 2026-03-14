import cv2
import numpy as np
import os

OUTPUT_DIR = "outputs/quality_overlay"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_blur_regions(gray):

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    variance = cv2.convertScaleAbs(laplacian)

    blur_mask = variance < 15

    return blur_mask


def detect_overexposed(gray):

    over_mask = gray > 240

    return over_mask


def generate_quality_overlay(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_mask = detect_blur_regions(gray)

    over_mask = detect_overexposed(gray)

    overlay = image.copy()

    # blur regions → gray
    overlay[blur_mask] = (120,120,120)

    # glare regions → yellow
    overlay[over_mask] = (0,255,255)

    output = cv2.addWeighted(image,0.7,overlay,0.3,0)

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"quality_{filename}"
    )

    cv2.imwrite(output_path,output)

    return {
        "quality_map": output_path
    }