# Training & Google Colab Deployment Guide

This directory provides everything needed to train the 5-class **Maize (Corn) Leaf Disease and Out-of-Domain Detection Model** on Google Colab using a free GPU.

---

## 🌟 What the New Model Does

The new training pipeline teaches the model to:
1. **Distinguish Maize vs. Non-Maize (`Not_Leaf`)**:
   - **Sugarcane & Grassy Monocot Leaves**: Accurately distinguishes sugarcane (*Saccharum*), sorghum, bamboo, and wild grasses from true maize (*Zea mays*), overcoming visual confusion caused by parallel venation.
   - **Written Documents & Handwriting**: Firmly rejects photos of notebook pages, handwriting, printed text, math equations, books, and receipts.
   - **Flowers & Blooms**: Rejects floral petals, roses, sunflowers, orchids, etc.
   - **Other Crop Leaves**: Rejects tomato, potato, apple, grape, pepper, and non-corn foliage.
   - **Everyday Objects & Scenes**: Rejects indoor objects, animals, vehicles, and furniture.
   - Uses both direct 5-class classification and calibrated feature-space Mahalanobis distance gating (`ood_stats.npz`).
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
4. Run all cells or step-by-step:
   - **Step 1**: Verifies NVIDIA GPU and installs dependencies.
   - **Step 2**: Writes `train.py` and `fit_ood.py` scripts.
   - **Step 3 & 4**: Configures datasets and initializes synthetic text/document & object generators.
   - **Step 5**: Assembles Maize classes (Blight, Rust, Gray Spot, Healthy) and downloads/synthesizes **Sugarcane**, **Written Documents**, **Flowers**, and **Other Crops** into `Not_Leaf`.
   - **Step 6**: Verifies balanced class distribution across all 5 classes.
   - **Step 7**: Two-stage MobileNetV2 transfer learning (frozen head stage + fine-tuning stage).
   - **Step 8**: Evaluates on held-out test split, checking non-maize rejection and plotting confusion matrix.
   - **Step 9**: Automatically packages and downloads `leafcheck_model.zip`.

---

### Option 2: Running via Colab Command Line / Python Script

If you prefer running a single script in Colab or on a remote GPU server:

1. Upload `training/train_colab.py` to Colab.
2. In a Colab cell, run:
   ```bash
   !pip install -q tensorflow keras scikit-learn pillow matplotlib kagglehub
   !python train_colab.py --data /content/dataset --auto-negative --out /content/output
   ```
3. Download the generated `/content/output/model_export.zip`.

---

## 📦 How to Deploy the Trained Model to Your Local App

Once `leafcheck_model.zip` (or `model_export.zip`) downloads to your computer:

1. Unzip the archive. It contains:
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
   - Any sugarcane leaf, non-corn crop, flower, written document, or everyday object will be rejected and show **⚠️ Crop: Not Maize (Corn)** and **Not a Maize Leaf**.

---

## 🏷️ Class Mapping Reference

| Canonical Class | Common Dataset Folder Names Automatically Recognized |
|---|---|
| `Blight` | `Northern_Leaf_Blight`, `Corn___Northern_Leaf_Blight`, `Corn_(maize)___Northern_Leaf_Blight` |
| `Common_Rust` | `Common_Rust`, `Corn___Common_rust`, `Corn_(maize)___Common_rust_` |
| `Gray_Leaf_Spot` | `Gray_Leaf_Spot`, `Corn___Cercospora_leaf_spot`, `Cercospora` |
| `Healthy` | `Healthy`, `Corn___healthy`, `Corn_(maize)___healthy` |
| `Not_Leaf` | `sugarcane`, `sorghum`, `bamboo`, `rice`, `wheat`, `grass`, other crops (`tomato`, `potato`, `apple`, etc.), flowers (`rose`, `sunflower`), documents/writing (`document`, `text`, `paper`, `writing`), or objects (`imagenet`, `cifar`, `car`, `indoor`) |
