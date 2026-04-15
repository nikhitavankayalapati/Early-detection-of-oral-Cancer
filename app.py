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
torch.set_num_threads(1)  # 🔥 important

# ================= APP =================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cpu")

# ================= LOAD MODELS =================
print("🚀 Loading models...")

# ---------- CNN ----------
cnn = None
try:
    cnn = models.mobilenet_v2(weights=None)   # 🔥 lighter than EfficientNet
    cnn.classifier[1] = nn.Linear(cnn.classifier[1].in_features, 2)

    checkpoint = torch.load("models/oral_cancer_cnn (2).pth", map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        cnn.load_state_dict(checkpoint["model_state_dict"])
    else:
        cnn.load_state_dict(checkpoint)

    cnn.to(DEVICE)
    cnn.eval()

    print("✅ CNN model loaded")

except Exception as e:
    print("❌ CNN load error:", e)

# ---------- XGBOOST ----------
xgb = None
try:
    xgb = joblib.load("models/oral_cancer_metadata_xgb (2).pkl")
    print("✅ XGB model loaded")
except Exception as e:
    print("❌ XGB load error:", e)

# ---------- FUSION ----------
IMAGE_WEIGHT = 0.75
META_WEIGHT = 0.25
THRESHOLD = 0.6

# ================= TRANSFORM =================
img_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 🔥 faster
    transforms.ToTensor()
])

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_file_name = None
    error = None

    if request.method == "POST":
        try:
            start_time = time.time()

            # ---------- IMAGE ----------
            image_file = request.files.get("image")

            if not image_file or image_file.filename == "":
                return render_template("index.html", error="Upload image")

            image_file_name = str(int(time.time())) + "_" + image_file.filename
            image_path = os.path.join(UPLOAD_FOLDER, image_file_name)
            image_file.save(image_path)

            image = Image.open(image_path).convert("RGB")
            image = img_transform(image).unsqueeze(0).to(DEVICE)

            # ---------- CNN ----------
            img_prob = 0.5
            if cnn is not None:
                try:
                    cnn_start = time.time()

                    with torch.no_grad():
                        logits = cnn(image)
                        probs = torch.softmax(logits, dim=1)[0]
                        img_prob = probs[1].item()

                    cnn_time = time.time() - cnn_start
                    print("⏱ CNN:", cnn_time)

                    # ⏱ timeout safety
                    if cnn_time > 6:
                        print("⚠️ CNN slow → fallback")
                        img_prob = 0.5

                except Exception as e:
                    print("❌ CNN error:", e)

            # ---------- METADATA ----------
            age = float(request.form.get("age", 0))
            gender = 1 if request.form.get("gender") == "M" else 0
            smoking = 1 if request.form.get("smoking") == "Yes" else 0
            chewing = 1 if request.form.get("chewing") == "Yes" else 0
            alcohol = 1 if request.form.get("alcohol") == "Yes" else 0

            age_norm = min(age / 100.0, 1.0)

            meta_prob = 0.5
            if xgb is not None:
                meta = np.array([[age_norm, gender, smoking, chewing, alcohol]])
                meta_prob = xgb.predict_proba(meta)[0][1]

            # ---------- FUSION ----------
            final_prob = (IMAGE_WEIGHT * img_prob) + (META_WEIGHT * meta_prob)

            result = "OPMD Detected" if final_prob >= THRESHOLD else "Healthy"
            confidence = round(final_prob * 100, 2)

            print("⏱ TOTAL:", time.time() - start_time)

        except Exception as e:
            error = str(e)
            print("❌ ERROR:", e)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_file=image_file_name,
        error=error
    )

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
