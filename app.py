from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFile
import joblib
import numpy as np
import os
import time

# ================= SAFETY =================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= APP =================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cpu")

# ================= CNN MODEL =================
cnn = models.efficientnet_b0(weights=None)
cnn.classifier[1] = nn.Linear(cnn.classifier[1].in_features, 2)

checkpoint = torch.load(
    "models/oral_cancer_cnn (2).pth",
    map_location=DEVICE
)

cnn.load_state_dict(checkpoint["model_state_dict"])
cnn.to(DEVICE)
cnn.eval()

# ================= XGBOOST =================
xgb = joblib.load("models/oral_cancer_metadata_xgb (2).pkl")

# ================= FUSION =================
fusion = joblib.load("models/fusion_config (2).pkl")

IMAGE_WEIGHT = fusion.get("image_weight", 0.75)
META_WEIGHT = fusion.get("meta_weight", 0.25)
THRESHOLD = fusion.get("threshold", 0.60)

# ================= TRANSFORM =================
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_file_name = None

    if request.method == "POST":

        # ---------- IMAGE ----------
        image_file = request.files.get("image")

        if not image_file or image_file.filename == "":
            return render_template("index.html", error="Please upload an image")

        # unique filename
        image_file_name = str(int(time.time())) + "_" + image_file.filename
        image_path = os.path.join(UPLOAD_FOLDER, image_file_name)
        image_file.save(image_path)

        image = Image.open(image_path).convert("RGB")
        image = img_transform(image).unsqueeze(0).to(DEVICE)

        # ---------- CNN ----------
        with torch.no_grad():
            logits = cnn(image)
            probs = torch.softmax(logits, dim=1)[0]

        # Assume class 1 = OPMD
        img_prob = probs[1].item()

        # ---------- METADATA ----------
        age = float(request.form["age"])
        gender = 1 if request.form["gender"] == "M" else 0
        smoking = 1 if request.form["smoking"] == "Yes" else 0
        chewing = 1 if request.form["chewing"] == "Yes" else 0
        alcohol = 1 if request.form["alcohol"] == "Yes" else 0

        age_norm = min(age / 100.0, 1.0)

        meta = np.array([[age_norm, gender, smoking, chewing, alcohol]])
        meta_prob = xgb.predict_proba(meta)[0, 1]

        # ---------- FUSION ----------
        final_prob = (IMAGE_WEIGHT * img_prob) + (META_WEIGHT * meta_prob)

        result = "OPMD Detected" if final_prob >= THRESHOLD else "Healthy"
        confidence = round(final_prob * 100, 2)

        # ---------- DEBUG ----------
        print("CNN:", img_prob)
        print("META:", meta_prob)
        print("FINAL:", final_prob)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_file=image_file_name
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)