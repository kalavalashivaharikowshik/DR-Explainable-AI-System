"""Evaluate the trained model on data/processed/test.

    python -m src.models.classification.evaluate
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config.config import CLASS_NAMES, MODEL_PATH
from src.models.classification.efficientnet_model import get_model
from src.models.loaders.dataset_loader import get_dataloaders
from src.utils.metrics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders()

    model = get_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    labels_all, preds_all = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            preds_all.extend(model(images.to(device)).argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())

    labels_all, preds_all = np.array(labels_all), np.array(preds_all)
    m = summarize(labels_all, preds_all)

    print("\nConfusion matrix (rows = true, cols = predicted)")
    print(confusion_matrix(labels_all, preds_all, labels=range(len(CLASS_NAMES))))
    print("\nClassification report")
    print(classification_report(labels_all, preds_all, labels=range(len(CLASS_NAMES)),
                                target_names=CLASS_NAMES, zero_division=0))
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"Quadratic weighted kappa: {m['qwk']:.4f}")


if __name__ == "__main__":
    main()
