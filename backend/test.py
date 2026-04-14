import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import build_model

# ---------------- CONFIG ----------------
DATA_DIR = "data/test"
MODEL_PATH = "model.pth"
BATCH_SIZE = 32

class_names = ["dress", "jeans", "shirt", "shoes"]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------- LOAD DATA ----------------
test_data = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", test_data.classes)

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------- TEST LOOP ----------------
correct = 0
total = 0

# Per-class stats
class_correct = [0] * len(class_names)
class_total = [0] * len(class_names)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        for i in range(len(labels)):
            label = labels[i].item()
            pred = predicted[i].item()

            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1

# ---------------- RESULTS ----------------
accuracy = 100 * correct / total
print(f"\nOverall Accuracy: {accuracy:.2f}%")

print("\nClass-wise Accuracy:")
for i in range(len(class_names)):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{class_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"{class_names[i]}: No samples")