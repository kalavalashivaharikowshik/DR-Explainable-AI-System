import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.classification.efficientnet_model import get_model

# ==============================
# SETTINGS
# ==============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/trained/best_model.pth"

OUTPUT_DIR = "outputs/heatmaps"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD MODEL
# ==============================

model = get_model().to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
)
model.eval()

# EfficientNet final convolution layer
target_layer = model.features[-1]

# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==============================
# GRADCAM CLASS
# ==============================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, class_idx=None):

        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        output[0, class_idx].backward()

        gradients = self.gradients
        activations = self.activations

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        activations = activations.squeeze(0)

        for i in range(pooled_gradients.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()

        heatmap = np.maximum(heatmap, 0)

        heatmap /= np.max(heatmap)

        return heatmap, class_idx


gradcam = GradCAM(model, target_layer)

# ==============================
# MAIN FUNCTION
# ==============================

def run_gradcam(image_path):

    image = Image.open(image_path).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    heatmap, class_idx = gradcam.generate(img_tensor)

    prediction = CLASS_NAMES[class_idx]

    # Load original image
    original = cv2.imread(image_path)

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # Save result
    filename = os.path.basename(image_path)

    output_path = os.path.join(OUTPUT_DIR, f"gradcam_{filename}")

    cv2.imwrite(output_path, overlay)

    return {
        "prediction": prediction,
        "heatmap_path": output_path
    }