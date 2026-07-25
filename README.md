# AI-Assisted Multimodal Deep Learning Framework for Early Detection of Oral Cancer

## Overview

This project presents an AI-powered multimodal framework for the early detection of Oral Potentially Malignant Disorders (OPMD) and Oral Squamous Cell Carcinoma (OSCC). Unlike traditional image-only approaches, the system combines oral cavity images with patient clinical metadata to improve diagnostic accuracy and support early clinical decision-making.

The framework employs transfer learning with multiple pre-trained CNN models and a machine learning classifier for patient metadata. The final prediction is generated using an ensemble fusion strategy that closely mimics real-world clinical diagnosis.

---

## Features

- Deep learning-based oral lesion classification
- Multimodal learning using images and patient metadata
- Comparison of multiple pre-trained CNN architectures
- Ensemble learning for improved prediction accuracy
- Real-time prediction through a Flask web application
- Confidence score for each prediction
- User-friendly web interface

---

## Technologies Used

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras

### Machine Learning
- XGBoost
- Scikit-learn

### Image Processing
- OpenCV
- NumPy

### Data Visualization
- Matplotlib
- Pandas

### Web Framework
- Flask

---

## Deep Learning Models Evaluated

- EfficientNet-B0
- MobileNetV3-Large
- ResNet-50
- DenseNet-121
- InceptionV3

---

## Dataset

The project uses the **Annotated Oral Cavity Images for Oral Cancer Detection** dataset obtained from **Zenodo**.

Dataset contains:

- Oral cavity images
- Healthy samples
- OPMD samples
- Oral Cancer samples
- Patient metadata
  - Age
  - Gender
  - Smoking habit
  - Alcohol consumption
  - Betel quid chewing

---

## Project Workflow

1. Dataset Collection
2. Image Preprocessing
3. Metadata Preprocessing
4. CNN Feature Extraction
5. Metadata Classification
6. Ensemble Fusion
7. Threshold Optimization
8. Prediction
9. Web Deployment using Flask

---

## Model Performance

| Model | Validation Accuracy |
|-------|--------------------:|
| MobileNetV3-Large | 77.51% |
| ResNet-50 | 78.57% |
| DenseNet-121 | 75.66% |
| EfficientNet-B0 | 79.37% |
| **Multimodal Ensemble Model** | **83.33%** |

Additional Metrics

- Accuracy: **83.33%**
- F1 Score: **88.9%**
- Matthews Correlation Coefficient (MCC): **0.56**

---

## Folder Structure

```
project/
│
├── dataset/
│   ├── images/
│   ├── metadata.csv
│
├── models/
│   ├── efficientnet_model.h5
│   ├── xgboost_model.pkl
│
├── static/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/oral-cancer-detection.git
```

Move into the project folder

```bash
cd oral-cancer-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## Results

The proposed multimodal framework successfully improves early oral cancer detection by combining visual features with patient clinical information.

Compared to image-only models, the ensemble model achieved higher accuracy and more reliable predictions for Oral Potentially Malignant Disorders.

---

## Future Improvements

- Explainable AI (Grad-CAM, Attention Maps)
- Multi-class oral disease classification
- Mobile application deployment
- Cloud deployment
- Telemedicine integration
- Larger and more diverse datasets

---

## Authors

- Amrutasagar Kavarthapu
- Vankayalapati Nikhita
- Naragani Nivas
- Puritipati Manasa
- Purilla Mahidhar

Department of CSE (AI & ML)

Seshadri Rao Gudlavalleru Engineering College

---

## License

This project is developed for academic and research purposes.
