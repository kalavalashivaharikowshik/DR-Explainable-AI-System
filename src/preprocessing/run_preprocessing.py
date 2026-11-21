"""Build the training dataset: dr_unified_v2/<split>/<class>/*  ->  data/processed/...

Usage:
    python -m src.preprocessing.run_preprocessing --input dr_unified_v2 --output data/processed
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import cv2
from tqdm import tqdm

from src.config.config import MODEL_INPUT_SIZE
from src.preprocessing.resize_normalize import preprocess_for_training

SPLITS = ["train", "val", "test"]
CLASSES = ["0", "1", "2", "3", "4"]


def process_image(args):
    in_path, out_path = args

    try:
        img = cv2.imread(in_path)
        if img is None:
            return f"unreadable: {in_path}"

        img = preprocess_for_training(img, MODEL_INPUT_SIZE)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
        return None

    except Exception as e:  # keep going, but report what failed and why
        return f"error: {in_path}: {e}"


def collect_tasks(input_dir, output_dir):
    tasks = []

    for split in SPLITS:
        for cls in CLASSES:
            class_dir = os.path.join(input_dir, split, cls)

            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                tasks.append((
                    os.path.join(class_dir, img_name),
                    os.path.join(output_dir, split, cls, img_name),
                ))

    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dr_unified_v2")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    print("\nScanning dataset...")
    tasks = collect_tasks(args.input, args.output)
    print(f"Total images found: {len(tasks)}")

    num_workers = max(1, cpu_count() - 2)
    failures = []

    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(process_image, tasks), total=len(tasks), desc="Processing images"):
            if result:
                failures.append(result)

    print(f"\nPreprocessing completed. {len(tasks) - len(failures)} ok, {len(failures)} failed.")
    for f in failures[:20]:
        print("  ", f)


if __name__ == "__main__":
    main()
