import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

from src.models.loaders.dataset_loader import get_dataloaders
from src.models.classification.efficientnet_model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/trained/best_model.pth"

# Load data
_, _, test_loader = get_dataloaders()

# Load model
model = get_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Convert to numpy
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds)

kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

print("\n📊 Confusion Matrix")
print(cm)

print("\n📄 Classification Report")
print(report)

print(f"\n⭐ Quadratic Weighted Kappa: {kappa:.4f}")