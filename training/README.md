# Training & Google Colab Deployment Guide

This directory provides everything needed to train the 5-class **Maize (Corn) Leaf Disease and Out-of-Domain Detection Model** on Google Colab using a free GPU.

---

## 🌟 What the New Model Does

The new training pipeline teaches the model to:
1. **Distinguish Maize vs. Non-Maize**:
   - Accurately classifies whether the photo is a real maize leaf or a non-corn / out-of-domain image (`Not_Leaf`).
   - Uses both direct 5-class classification and feature-space Mahalanobis distance gating (`ood_stats.npz`).
2. **Detect Foliar Diseases with High Accuracy**:
   - **Northern Leaf Blight** (*Exserohilum turcicum*)
   - **Common Rust** (*Puccinia sorghi*)
   - **Gray Leaf Spot** (*Cercospora zeae-maydis*)
   - **Healthy Maize** (No disease detected)

---

## 🚀 How to Train on Google Colab (Step-by-Step)

### Option 1: Using the Interactive Colab Notebook (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File → Upload notebook** and upload:
   ```
   training/Leafcheck_Colab_Training.ipynb
   ```
3. Enable GPU acceleration:
   - Click **Runtime → Change runtime type**
   - Select **Hardware accelerator: T4 GPU**
   - Click **Save**
4. Run each cell step-by-step:
   - **Step 1**: Verifies NVIDIA GPU and installs dependencies.
   - **Step 2**: Downloads or mounts dataset (Kaggle or Google Drive).
   - **Step 3**: Automatically prepares the 5th class (`Not_Leaf` for non-maize / other plant leaves).
   - **Step 4**: Performs image normalization and MD5/dHash visual deduplication.
   - **Step 5**: Creates stratified 70% Train, 15% Validation, 15% Test split.
   - **Step 6**: Builds MobileNetV2 architecture with pretrained ImageNet weights.
   - **Step 7**: Two-stage training (Stage 1: Head training; Stage 2: Fine-tuning top 40 layers).
   - **Step 8**: Evaluates on test set and plots confusion matrix.
   - **Step 9**: Computes Mahalanobis Out-of-Distribution gate (`ood_stats.npz`).
   - **Step 10**: Packages and automatically downloads `model_export.zip`.

---

### Option 2: Running via Colab Command Line / Python Script

If you prefer running a single script in Colab or on a remote GPU server:

1. Upload `training/train_colab.py` to Colab.
2. In a Colab cell, run:
   ```bash
   !pip install -q tensorflow keras scikit-learn pillow matplotlib
   !python train_colab.py --data /content/dataset --auto-negative --out /content/output
   ```
3. Download the generated `/content/output/model_export.zip`.

---

## 📦 How to Deploy the Trained Model to Your Local App

Once `model_export.zip` downloads to your computer:

1. Unzip `model_export.zip`. It contains:
   - `corn_leaf_disease_model_v3.keras`
   - `class_names.json`
   - `ood_stats.npz`
   - `metrics.json`
   - `confusion_matrix.png`
2. Copy the first three files into your backend model directory:
   ```
   d:/final-year-project/backend/model/
   ```
3. Restart your local backend server:
   ```bash
   cd d:\final-year-project\backend
   .\.venv\Scripts\activate
   uvicorn app.main:app --reload --port 8000
   ```
4. Open the frontend at `http://localhost:3000/detect`.
   - Any corn leaf photo will show the explicit **Disease Name**, **Crop: Maize (Corn)**, **Causative Pathogen**, **Severity**, and **Explainability Heatmap**.
   - Any non-corn photo or invalid image will show **⚠️ Crop: Not Maize (Corn)** and **Not a Maize Leaf**.

---

## 🏷️ Class Mapping Reference

| Canonical Class | Common Dataset Folder Names Automatically Recognized |
|---|---|
| `Blight` | `Northern_Leaf_Blight`, `Corn___Northern_Leaf_Blight`, `Corn_(maize)___Northern_Leaf_Blight` |
| `Common_Rust` | `Common_Rust`, `Corn___Common_rust`, `Corn_(maize)___Common_rust_` |
| `Gray_Leaf_Spot` | `Gray_Leaf_Spot`, `Corn___Cercospora_leaf_spot`, `Cercospora` |
| `Healthy` | `Healthy`, `Corn___healthy`, `Corn_(maize)___healthy` |
| `Not_Leaf` | Folders with `not`, `non`, `other`, `negative`, `background`, or other crops (`tomato`, `potato`, `apple`, etc.) |
