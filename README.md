# Leafcheck: Corn Leaf Disease Detection

FastAPI backend + Next.js frontend around a MobileNetV2 architecture (`corn_leaf_disease_model_v3.keras`).
Features 5-class detection (Northern Leaf Blight, Common Rust, Gray Leaf Spot, Healthy Maize, and Not_Leaf / Non-Maize), out-of-domain rejection gating (`ood_stats.npz`), foliage validation, explainable Grad-CAM heatmaps, and actionable agronomic recommendations.

## Run locally (no Docker)

Backend (Python 3.11+):

    cd backend
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000

Uses SQLite by default (`backend/corn_leaf.db`). API docs: http://localhost:8000/docs

Frontend (Node 20+):

    cd frontend
    cp .env.example .env.local
    npm install
    npm run dev          # http://localhost:3000

## Run with Docker (PostgreSQL)

    cp .env.example .env      # set SECRET_KEY (openssl rand -hex 32) and POSTGRES_PASSWORD
    docker compose up --build

## API

| Method | Path | Notes |
|---|---|---|
| POST | /api/auth/register, /api/auth/login | JWT (login takes form fields `username`=email, `password`) |
| GET | /api/users/me | |
| POST | /api/predict | multipart `file`; JPEG/PNG/WebP/BMP, max 8 MB |
| GET | /api/predictions, /api/predictions/{id} | history, newest first |
| GET | /api/predictions/{id}/image, /heatmap | owner-only images |
| DELETE | /api/predictions/{id} | |
| GET | /api/stats | dashboard numbers |
| GET | /api/health | |

## Important details

- **Preprocessing:** the saved model has a `Rescaling(1/127.5, -1)` layer inside it, so the backend sends raw 0-255 pixels
  resized to 224x224. Do not add `preprocess_input` in front of it, or inputs get scaled twice.
- **Grad-CAM:** computed on the MobileNetV2 `out_relu` feature map (7x7x1280) using `tf.GradientTape`, upsampled and overlaid with OpenCV.
- **Only corn leaves get a diagnosis.** A photo is flagged with `"This doesn't look like a corn leaf"` (`is_corn_leaf: false`, crop status `Non-Maize / Unrecognized`) if any of these triggers:
  1. **Direct negative classification**: The model predicts `Not_Leaf` / Non-Maize (available in models trained with the 5th class; see `training/`);
  2. **Non-foliage verification**: Color and vegetation analysis detects insufficient green/plant hues (e.g. solid color backgrounds, random objects);
  3. **Feature-space OOD gate**: Distance to the corn-leaf feature statistics in `backend/model/ood_stats.npz` exceeds the threshold (`OOD_SCALE` parameter);
  4. **Low confidence**: Model prediction confidence falls below `REJECT_BELOW_CONFIDENCE` (default 0.55).
- **Google Colab Training**: Train the updated 5-class model with one click in Google Colab using `training/Leafcheck_Colab_Training.ipynb` or `training/train_colab.py`. See [training/README.md](file:///d:/final-year-project/training/README.md) for full steps.
- **Tests:** `cd backend && pytest` (runs against the real model).
- Set a strong `SECRET_KEY` and restrict `CORS_ORIGINS` before deploying.
