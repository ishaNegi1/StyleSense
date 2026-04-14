import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# ---------------- CONFIG ----------------
IMAGE_FOLDER = "images_compressed"
OUTPUT_DIR = "data"
TARGET_SIZE = (128, 128)

# ---------------- LOAD ----------------
df = pd.read_csv("images.csv")
df["label"] = df["label"].str.strip().str.lower()

# ---------------- STRICT FILTER ----------------
def map_label(label):
    if label in ["shirt", "t-shirt"]:
        return "shirt"
    elif label == "pants":
        return "jeans"
    elif label == "dress":
        return "dress"
    elif label == "shoes":
        return "shoes"
    else:
        return None

df["clean_label"] = df["label"].apply(map_label)
df = df.dropna(subset=["clean_label"])

print("\nAfter filtering:")
print(df["clean_label"].value_counts())

# ---------------- BALANCE ----------------
min_count = df["clean_label"].value_counts().min()

df = df.groupby("clean_label", group_keys=False).apply(
    lambda x: x.sample(n=min_count, random_state=42)
).reset_index(drop=True)

print("\nAfter balancing:")
print(df["clean_label"].value_counts())

# ---------------- SPLIT ----------------
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["clean_label"], random_state=42
)

# ---------------- CHECK FOLDER ----------------
if not os.path.exists(IMAGE_FOLDER):
    raise FileNotFoundError("❌ images_compressed not found")

# ---------------- CREATE DATASET ----------------
copied = 0
missing = 0

for split, split_df in [("train", train_df), ("test", test_df)]:
    for _, row in split_df.iterrows():
        label = row["clean_label"]
        filename = str(row["image"]).strip()

        src = None
        for ext in ["", ".jpg", ".jpeg", ".png", ".webp"]:
            path = os.path.join(IMAGE_FOLDER, filename + ext)
            if os.path.exists(path):
                src = path
                break

        dst_dir = os.path.join(OUTPUT_DIR, split, label)
        os.makedirs(dst_dir, exist_ok=True)

        if src:
            try:
                img = Image.open(src).convert("RGB")
                img = img.resize(TARGET_SIZE)

                save_path = os.path.join(dst_dir, filename + ".jpg")
                img.save(save_path, "JPEG", quality=85)

                copied += 1
            except Exception as e:
                print(f"Error: {filename} → {e}")
        else:
            missing += 1

print(f"\nDone → Copied: {copied}, Missing: {missing}")

# ---------------- VERIFY ----------------
print("\nFinal data dataset:")
for split in ["train", "test"]:
    for cls in ["dress", "jeans", "shirt", "shoes"]:
        path = os.path.join(OUTPUT_DIR, split, cls)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"{split}/{cls}: {count}")