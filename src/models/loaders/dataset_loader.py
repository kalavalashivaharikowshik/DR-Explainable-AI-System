import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==============================
# SETTINGS
# ==============================
from src.config.config import IMAGENET_MEAN, IMAGENET_STD, MODEL_INPUT_SIZE

DATA_DIR = os.environ.get("DR_DATA_DIR", "data/processed")
IMG_SIZE = MODEL_INPUT_SIZE
BATCH_SIZE = 16

# ==============================
# TRANSFORMS
# ==============================

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ==============================
# LOAD DATASETS
# ==============================

def get_dataloaders():

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=val_test_transforms
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform=val_test_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
