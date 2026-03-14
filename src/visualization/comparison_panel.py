import os
import random

REFERENCE_DIR = "data/reference_images"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]


def get_reference_images(prediction):

    class_index = CLASS_NAMES.index(prediction)

    same_class_dir = os.path.join(REFERENCE_DIR, str(class_index))

    same_examples = os.listdir(same_class_dir)

    same_image = random.choice(same_examples)

    same_path = os.path.join(same_class_dir, same_image)


    # also show one severe example if available
    severe_dir = os.path.join(REFERENCE_DIR, "4")

    severe_examples = os.listdir(severe_dir)

    severe_image = random.choice(severe_examples)

    severe_path = os.path.join(severe_dir, severe_image)


    return {
        "same_class_example": same_path,
        "severe_example": severe_path
    }