import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request
from PIL import Image
import joblib
import numpy as np

# =========================
# Flask App
# =========================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

torch.set_num_threads(1)

# =========================
# Load CNN (EfficientNet)
# =========================
def load_cnn():
    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, 2)
        model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
        model.eval()
        print("✅ CNN loaded")
        return model
    except Exception as e:
        print("❌ CNN error:", e)
        return None

cnn_model = load_cnn()

# =========================
# Load Fusion Model
# =========================
try:
    fusion_model = joblib.load("fusion_model.pkl")
    print("✅ Fusion model loaded")
except Exception as e:
    print("❌ Fusion load error:", e)
    fusion_model = None

# =========================
# Image Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# CNN Feature / Probability
# =========================
def get_cnn_prob(image_path):
    if cnn_model is None:
        return 0.5

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(image)
        prob = torch.softmax(output, dim=1)[0][1].item()

    return prob

# =========================
# Fusion Prediction
# =========================
def predict_fusion(image_path, age, gender, smoking, chewing, alcohol):
    if fusion_model is None:
        return "Error", 0

    # CNN output
    cnn_prob = get_cnn_prob(image_path)

    # Encode inputs
    gender = 1 if gender == "M" else 0
    smoking = 1 if smoking == "Yes" else 0
    chewing = 1 if chewing == "Yes" else 0
    alcohol = 1 if alcohol == "Yes" else 0

    # Final feature vector
    features = np.array([[cnn_prob, age, gender, smoking, chewing, alcohol]])

    # Prediction
    prob = fusion_model.predict_proba(features)[0][1]

    result = "Cancer Detected" if prob > 0.5 else "No Cancer"
    confidence = round(prob * 100, 2)

    return result, confidence

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_file = None
    error = None

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
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], image_file)
            file.save(filepath)

            # Predict
            result, confidence = predict_fusion(
                filepath, age, gender, smoking, chewing, alcohol
            )

        except Exception as e:
            error = str(e)
            print("❌ Error:", e)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_file=image_file,
        error=error
    )

# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)
