import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request
from PIL import Image
import joblib
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# DEVICE
# =========================
device = torch.device("cpu")

# =========================
# LOAD CNN MODEL (EfficientNet)
# =========================
print("🚀 Loading models...")

cnn_model = None
threshold = 0.5
image_weight = 0.5

try:
    cnn_path = os.path.join("models", "oral_cancer_cnn (2).pth")

    checkpoint = torch.load(cnn_path, map_location=device)

    # Load EfficientNet
    cnn_model = models.efficientnet_b0(weights=None)
    cnn_model.classifier[1] = nn.Linear(cnn_model.classifier[1].in_features, 1)

    # IMPORTANT FIX
    cnn_model.load_state_dict(checkpoint["model_state_dict"])

    cnn_model.eval()

    # Extract metadata
    threshold = checkpoint.get("threshold", 0.5)
    image_weight = checkpoint.get("image_weight", 0.5)

    print("✅ CNN loaded")
    print("Threshold:", threshold)
    print("Image weight:", image_weight)

except Exception as e:
    print("❌ CNN load error:", e)

# =========================
# LOAD XGB MODEL
# =========================
try:
    xgb_model = joblib.load(os.path.join("models", "oral_cancer_metadata_xgb (2).pkl"))
    print("✅ XGB loaded")
except Exception as e:
    print("❌ XGB error:", e)
    xgb_model = None

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_file = None

    if request.method == "POST":
        try:
            file = request.files["image"]
            age = int(request.form["age"])
            gender = request.form["gender"]
            smoking = request.form["smoking"]
            chewing = request.form["chewing"]
            alcohol = request.form["alcohol"]

            # Save image
            image_file = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, image_file)
            file.save(filepath)

            # =========================
            # CNN Prediction
            # =========================
            img = Image.open(filepath).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                cnn_output = cnn_model(img)
                cnn_prob = torch.sigmoid(cnn_output).item()

            # =========================
            # XGB Prediction
            # =========================
            gender = 1 if gender == "M" else 0
            smoking = 1 if smoking == "Yes" else 0
            chewing = 1 if chewing == "Yes" else 0
            alcohol = 1 if alcohol == "Yes" else 0

            meta = np.array([[age, gender, smoking, chewing, alcohol]])

            xgb_prob = xgb_model.predict_proba(meta)[0][1]

            # =========================
            # FUSION
            # =========================
            final_score = (image_weight * cnn_prob) + ((1 - image_weight) * xgb_prob)

            result = "Cancer Detected" if final_score > threshold else "No Cancer"
            confidence = round(final_score * 100, 2)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           image_file=image_file)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
