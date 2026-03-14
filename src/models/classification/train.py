import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.loaders.dataset_loader import get_dataloaders
from src.models.classification.efficientnet_model import get_model

# ==============================
# SETTINGS
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
LEARNING_RATE = 3e-4

MODEL_SAVE_PATH = "models/trained/dr_efficientnet_b3_best.pth"

# ==============================
# LOAD DATA
# ==============================
train_loader, val_loader, test_loader = get_dataloaders()

# ==============================
# LOAD MODEL
# ==============================
model = get_model().to(DEVICE)

# Resume training if model exists
if os.path.exists(MODEL_SAVE_PATH):
    print("🔁 Loading existing model weights...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_acc = 0.0

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):

    print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")

    # -------- TRAIN --------
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item())

    train_acc = 100. * correct / total

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validation", leave=False)

        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100. * correct / total

    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.3f} | Val Acc:   {val_acc:.2f}%")    

    # -------- SAVE BEST MODEL --------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best model saved!")

print("\n🎉 Training Complete!")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
