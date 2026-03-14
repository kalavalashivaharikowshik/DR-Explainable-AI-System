import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==============================
# SETTINGS
# ==============================
DATA_DIR = "data/processed"
IMG_SIZE = 300
BATCH_SIZE = 16

# ==============================
# TRANSFORMS
# ==============================

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
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
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
