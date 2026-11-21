"""Train EfficientNet-B3 on data/processed/{train,val}.

    python -m src.models.classification.train --epochs 30

Improvements over a plain loop:
* class-weighted loss (DR datasets are dominated by "No DR")
* best checkpoint chosen by quadratic weighted kappa (the standard DR metric)
* cosine learning-rate schedule, fixed seed, averaged losses
* resumes from the checkpoint at --save-path if it exists
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config.config import MODEL_PATH, NUM_CLASSES
from src.models.classification.efficientnet_model import get_model
from src.models.loaders.dataset_loader import get_dataloaders
from src.utils.metrics import summarize


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(dataset) -> torch.Tensor:
    counts = np.bincount(dataset.targets, minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    training = optimizer is not None
    model.train(training)
    total_loss, labels_all, preds_all = 0.0, [], []

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            labels_all.extend(labels.cpu().numpy())
            preds_all.extend(outputs.argmax(1).cpu().numpy())

    metrics = summarize(labels_all, preds_all)
    metrics["loss"] = total_loss / max(len(labels_all), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders()

    model = get_model(pretrained=True).to(device)
    if os.path.exists(args.save_path):
        print(f"Resuming from {args.save_path}")
        model.load_state_dict(torch.load(args.save_path, map_location=device, weights_only=True))

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_loader.dataset).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_qwk = -1.0
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, optimizer, desc=f"train {epoch}")
        val_m = run_epoch(model, val_loader, criterion, device, desc=f"val {epoch}")
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {train_m['loss']:.3f} acc {train_m['accuracy']:.3f} | "
              f"val loss {val_m['loss']:.3f} acc {val_m['accuracy']:.3f} QWK {val_m['qwk']:.4f}")

        if val_m["qwk"] > best_qwk:
            best_qwk = val_m["qwk"]
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  saved new best model (QWK {best_qwk:.4f}) -> {args.save_path}")

    print(f"\nTraining complete. Best validation QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
