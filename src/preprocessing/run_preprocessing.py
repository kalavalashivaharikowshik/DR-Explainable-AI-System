import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==============================
# PATH SETTINGS
# ==============================
INPUT_DIR = "dr_unified_v2"
OUTPUT_DIR = "data/processed"
IMG_SIZE = 300

# ==============================
# PREPROCESS FUNCTIONS
# ==============================

def crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 10))
    
    if coords.shape[0] == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    return img[y0:y1, x0:x1]


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def ben_graham_preprocess(img):
    return cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), 10),
        -4, 128
    )


# ==============================
# MAIN PROCESS FUNCTION
# ==============================

def process_image(args):
    in_path, out_path = args

    try:
        img = cv2.imread(in_path)
        if img is None:
            return

        img = crop_black_borders(img)
        img = apply_clahe(img)
        img = ben_graham_preprocess(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)

    except Exception as e:
        print(f"Error processing: {in_path}")


# ==============================
# COLLECT ALL IMAGE PATHS
# ==============================

def collect_tasks():
    tasks = []

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(INPUT_DIR, split)

        for cls in ["0", "1", "2", "3", "4"]:
            class_dir = os.path.join(split_dir, cls)

            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                in_path = os.path.join(class_dir, img_name)

                out_path = os.path.join(
                    OUTPUT_DIR, split, cls, img_name
                )

                tasks.append((in_path, out_path))

    return tasks


# ==============================
# RUN PIPELINE
# ==============================

if __name__ == "__main__":
    print("\n🔍 Scanning dataset...")

    tasks = collect_tasks()
    total = len(tasks)

    print(f"📊 Total images found: {total}")
    print("🚀 Starting preprocessing...\n")

    num_workers = max(1, cpu_count() - 2)

    with Pool(num_workers) as pool:
        list(
            tqdm(
                pool.imap(process_image, tasks),
                total=total,
                desc="Processing Images",
                colour="green"
            )
        )

    print("\n✅ Preprocessing completed!")
