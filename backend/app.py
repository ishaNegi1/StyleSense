from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import os

from model import build_model

app = Flask(__name__)

# ✅ Simple CORS (just works everywhere)
CORS(app)

# ---------------- CLASS NAMES ----------------
class_names = ["dress", "jeans", "shirt", "shoes"]

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LAZY MODEL ----------------
_model = None

def get_model():
    global _model
    if _model is None:
        print("⏳ Loading model...")
        _model = build_model()
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        _model.load_state_dict(torch.load(model_path, map_location=device))
        _model = _model.to(device)
        _model.eval()
        print("✅ Model loaded successfully")
    return _model

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------- COLOR DETECTION ----------------
def detect_color(image):
    img = np.array(image)
    avg = img.mean(axis=(0,1))
    r, g, b = avg

    if r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif abs(r-g) < 25 and abs(g-b) < 25:
        return "gray"
    elif r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    else:
        return "mixed"

# ---------------- ROUTES ----------------
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    # ✅ Handle preflight (browser CORS)
    if request.method == "OPTIONS":
        return '', 200

    print("📥 Incoming request")

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            model = get_model()
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)

        category = class_names[predicted.item()]
        color = detect_color(image)

        return jsonify({
            "category": category,
            "confidence": round(float(confidence.item()), 3),
            "color": color
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- HEALTH ----------------
@app.route("/")
def home():
    return "✅ Backend is running"

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)