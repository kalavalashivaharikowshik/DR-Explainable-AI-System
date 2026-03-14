import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.classification.efficientnet_model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/trained/best_model.pth"

OUTPUT_DIR = "outputs/severity_bars"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]

# Load model
model = get_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


def generate_severity_bar(image_path):

    image = Image.open(image_path).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(img_tensor)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    prediction_idx = np.argmax(probs)
    prediction = CLASS_NAMES[prediction_idx]

    # create progress bar image
    bar_img = np.ones((150,800,3),dtype=np.uint8) * 255

    start_x = 50
    bar_width = 120

    for i in range(5):

        x = start_x + i * bar_width

        if i == prediction_idx:
            color = (0,0,255)
        else:
            color = (200,200,200)

        cv2.rectangle(
            bar_img,
            (x,70),
            (x+100,100),
            color,
            -1
        )

        cv2.putText(
            bar_img,
            CLASS_NAMES[i],
            (x,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            2
        )

    cv2.putText(
        bar_img,
        f"Prediction: {prediction}",
        (50,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,0),
        2
    )

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"severity_{filename}"
    )

    cv2.imwrite(output_path, bar_img)

    return {
        "prediction": prediction,
        "probabilities": probs.tolist(),
        "output_path": output_path
    }