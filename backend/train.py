import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from model import build_model

# ---------------- SETTINGS ----------------
DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 8
LR = 0.001

# ---------------- CHECK ----------------
print("Train exists:", os.path.exists(f"{DATA_DIR}/train"))
print("Test exists:", os.path.exists(f"{DATA_DIR}/test"))

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------- DATA ----------------
train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
test_data  = datasets.ImageFolder(f"{DATA_DIR}/test", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)

print("Classes:", train_data.classes)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- MODEL ----------------
model = build_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

# ---------------- TRAIN ----------------
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # -------- VALIDATION --------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} → Loss: {total_loss:.4f} → Acc: {acc:.1f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model2.pth")
        print("  ✅ Saved best model")

print(f"\nFinal Accuracy: {best_acc:.1f}%")

# ---------------- FINE-TUNING ----------------
print("\n🔥 Starting fine-tuning...")

# Unfreeze entire model
for param in model.parameters():
    param.requires_grad = True

# Lower learning rate (IMPORTANT)
optimizer = optim.Adam(model.parameters(), lr=0.00003)

# Short fine-tune
for epoch in range(3):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Fine-tune Epoch {epoch+1} → Loss: {total_loss:.4f} → Acc: {acc:.1f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model.pth")
        print("  ✅ Saved improved model")

print(f"\n🏆 Final Accuracy after fine-tuning: {best_acc:.1f}%")