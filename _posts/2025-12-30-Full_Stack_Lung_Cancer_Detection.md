---
title: "🫁 Lung Cancer Detection: CNN → Hugging Face Space → React UI → CI/CD"
date: 2025-12-30
permalink: /posts/2025-12-30-Full_Stack_Lung_Cancer_Detection.md/
tags:
  - Deep Learning
  - FastAPI
  - Docker
  - React
  - CI/CD
  - HuggingFace
  - Gradio
---

Lung cancer is one of the leading causes of cancer-related mortality worldwide. Early and accurate diagnosis plays a critical role in treatment planning and patient outcomes. Among the different diagnostic modalities, **histopathological analysis of lung tissue** remains a gold standard for confirming cancer type and subtype.

From a clinical perspective, lung cancer is commonly categorized into multiple histological subtypes. In this project, the focus is on three clinically relevant categories:

* **Normal lung tissue (`lung_n`)**
* **Lung adenocarcinoma (`lung_aca`)**
* **Lung squamous cell carcinoma (`lung_scc`)**

![Lung Cancer](/images/lung_cancer.webp)

(a) representing lung **adenocarcinoma**, <br>
(b) showing **lung squamous cell carcinoma**, <br>
(c) depicting **normal cells**.

Adenocarcinoma and squamous cell carcinoma are two major forms of non-small cell lung cancer (NSCLC), each associated with different growth patterns, treatment strategies, and prognostic implications. Distinguishing between these subtypes using histopathology images requires careful analysis of **cellular morphology, tissue architecture, and staining patterns**, which can be subtle and highly variable.

Ref: https://doi.org/10.1038/s41598-024-61101-7

---

## Why Convolutional Neural Networks (CNNs)?

Histopathology images are high-dimensional and spatially complex. Traditional machine learning approaches require handcrafted features, which often fail to generalize across staining variations and tissue heterogeneity.

**Convolutional Neural Networks (CNNs)** are well-suited for this task because they:

* Learn hierarchical visual features directly from raw images
* Capture local textures (e.g., nuclei shape, gland structure) and global morphology
* Are robust to small spatial variations through pooling operations

Early convolutional layers typically learn low-level patterns such as edges and color gradients, while deeper layers capture higher-level morphological features that differentiate cancer subtypes.

---

## Problem formulation

From a machine learning perspective, this task is formulated as a **multi-class image classification problem**:

* **Input:** RGB histopathology image of lung tissue
* **Output:** One of three class labels (`lung_n`, `lung_aca`, `lung_scc`) with associated probabilities

Performance evaluation goes beyond overall accuracy. In medical settings, **class-wise recall (sensitivity)** and **false negative rate (FNR)** are particularly important, as missed cancer cases can have serious clinical consequences.

---

## Motivation for an end-to-end system

Many academic projects stop at model training and offline evaluation. However, deploying a trained medical imaging model introduces additional challenges:

* Large model artifacts (often >100 MB)
* Reproducible inference environments
* Safe and consistent preprocessing
* Practical access through APIs or user interfaces

The goal of this project is not only to train a CNN, but to **build a complete, reproducible, end-to-end system** that takes a histopathology image as input and produces a clinically interpretable prediction through a live web interface.

The sections that follow document this process step by step—from dataset preparation and model training to deployment, containerization, and frontend integration—so the same workflow can be reused for similar medical imaging applications.

---

## System overview

The system consists of the following components:

* A **CNN-based image classifier** trained with TensorFlow/Keras
* **Saved model artifacts** (>100 MB)
* A **FastAPI inference service**
* A **Gradio-based Hugging Face Space** for free model hosting
* A **React frontend** built with Vite and Tailwind CSS
* **Docker** for reproducible backend deployment
* **GitHub Actions** for automated builds and deployment

---

## High-level architecture

The final setup supports two inference paths:

1. **Hugging Face Space inference**

   * The trained model is hosted on the Hugging Face Hub
   * A Gradio app loads the model and exposes a prediction function
   * The frontend sends images directly to the Space

2. **FastAPI inference**

   * A REST API loads the same trained model
   * Exposes `/predict` and `/health` endpoints
   * Can be run locally or inside Docker

This dual approach allows:

* free public inference without hosting a server
* a clean API implementation for local testing and containerization

---

## 1) Starting point: define the goal and freeze the dataset layout

I began with one clear objective: **classify lung histopathology images into 3 classes**:

* `lung_n` (normal)
* `lung_aca` (adenocarcinoma)
* `lung_scc` (squamous cell carcinoma)

The first practical step was ensuring a stable dataset layout that my training and inference code could rely on:

```
data/raw/lung_colon_image_set/lung_image_sets/
  lung_aca/
  lung_n/
  lung_scc/
```

This folder naming is important because later:

* training maps folders → class indices
* inference uses `classes.json` to map indices → labels 

Dataset Link: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

Typical setup commands:

```bash
mkdir -p data/raw
unzip lung_colon_image_set.zip -d data/raw/lung_colon_image_set
```

---

## 2) Phase 1 — Train the CNN model (only ML first)

I trained the CNN using the scripts under `src/lung_cancer/`:

* `dataset.py` (load + preprocess)
* `model.py` (CNN architecture)
* `train.py` (training loop)
* `evaluate.py` (classification report) 

Run training (from repo root):

```bash
python -m src.lung_cancer.train
```

Run evaluation:

```bash
python -m src.lung_cancer.evaluate
```

The training output is versioned under:

```
artifacts/models/v1/
  lung_cnn.keras
  classes.json
  metadata.json
artifacts/reports/
  classification_report.txt
```

This artifact layout is what the FastAPI backend loads later. 

---

## 3) Phase 2 — Test the trained model through FastAPI

Before building a UI, I validated inference through an API.

Your backend is in `backend/` and exposes:

* `GET /health`
* `POST /predict` (multipart image upload)

It loads model artifacts at startup using `ensure_model_files()` and creates `LungCancerPredictor`. 

### Run FastAPI locally

```bash
pip install -r backend/requirements-backend.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Predict using an image:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/sample_requests/example.png"
```

At this point, I had an end-to-end system working locally:
**image → API → model → JSON response**. 

---

## 4) Phase 3 — Dockerize the backend

Next, I containerized the backend using the repo `Dockerfile`.

Key details:

* CPU-only inference is enforced via `CUDA_VISIBLE_DEVICES=-1`
* backend dependencies come from `backend/requirements-backend.txt`
* code is copied from `src/` and `backend/`
* Uvicorn is the container entrypoint

### Build and run locally

```bash
docker build -t lung-cancer-api .
docker run --rm -p 8000:8000 lung-cancer-api
```

Test:

```bash
curl http://localhost:8000/health
```

---

## 5) Phase 4 — Push the Docker image to Docker Hub (pull-and-run workflow)

Once Docker worked locally, I published it so anyone can run:

### Login

```bash
docker login
```

### Tag the image

```bash
docker tag lung-cancer-api:latest mdislammazharul/lung-cancer-api:latest
```

### Push

```bash
docker push mdislammazharul/lung-cancer-api:latest
```

### Anyone can now pull and run it

```bash
docker pull mdislammazharul/lung-cancer-api:latest

docker run --rm -p 8000:8000 \
  -e ALLOWED_ORIGINS="*" \
  mdislammazharul/lung-cancer-api:latest
```

This gives a clean “one command to run inference locally” workflow.

---

## 6) Phase 5 — Handle the >100MB model file

After training, the model was **>100MB**, which created problems when I tried to push it normally to GitHub.

So I used two strategies (both useful depending on the situation):

### Option A: GitHub with Git LFS

One-time setup:

```bash
git lfs install
```

Track model files:

```bash
git lfs track "*.h5"
git lfs track "*.keras"
```

Commit + push:

```bash
git add .gitattributes
git add artifacts/models/v1/lung_cnn.keras
git commit -m "Track model with Git LFS"
git push origin main
```

### Option B: Hugging Face Hub (best for free deployment)

This project’s Hugging Face Space downloads the model from the Hub at runtime using `hf_hub_download(...)`. 

Install and login:

```bash
pip install -U huggingface_hub
huggingface-cli login
```

Clone your HF model repo:

```bash
git clone https://huggingface.co/mdislammazharul/Lung_Cancer_Detection
cd Lung_Cancer_Detection
```

Copy artifacts in:

```bash
cp /path/to/lung_cnn.h5 .
cp /path/to/classes.json .
```

Commit + push:

```bash
git add .
git commit -m "Upload trained model + classes mapping"
git push
```

---

## 7) Phase 6 — Deploy inference on Hugging Face Spaces (Gradio)

After the model artifacts were hosted, I deployed a free public inference app using Gradio.

The Space app (`Lung_Cancer_Detection_HF_Space/app.py`) does this:

* Forces CPU inference
* Downloads model + `classes.json` from HF Hub
* Loads the model with a compatibility patch (`DenseCompat`)
* Runs preprocessing (OpenCV resize, normalize)
* Returns JSON probabilities + `predicted_class` 

### Space deployment steps

1. Create a new Space on Hugging Face (Gradio)
2. Upload/push the contents of:

```
Lung_Cancer_Detection_HF_Space/
  app.py
  requirements.txt
```

Now the model is usable publicly without shipping the >100MB file inside the Space repo. 

---

## 8) Phase 7 — Connect via Gradio API (frontend-independent testing)

Before building the React UI, I confirmed I could call the Space programmatically.

```bash
pip install gradio_client
```

```python
from gradio_client import Client

client = Client("mdislammazharul/lung-cancer-detection-hf-space")
result = client.predict("data/sample_requests/example.png", fn_index=2)
print(result)
```

---

## 9) Phase 8 — Build the React frontend

Only after inference was stable (FastAPI + Space), I started to build the UI.

Frontend structure:

* `frontend/` (Vite + React)
* Tailwind styling
* Components like `ModelArchitecture.jsx`, `ModelPerformance.jsx`, `PredictionResult.jsx`
* API client: `frontend/src/lungSpaceApi.js` (can target local FastAPI or HF Space)

Run locally:

```bash
cd frontend
npm ci
npm run dev
```

At this stage I could:

* upload an image
* get predictions
* show metrics and model details on the same site

---

## 10) Phase 9 — CI/CD with GitHub Actions

After everything worked locally, I automated:

### A) CI: backend import smoke test

The `ci.yml` checks that the backend imports cleanly:

```bash
python -c "from backend.main import app; print('FastAPI import OK')"
```

This is a lightweight sanity check that catches broken imports early. 

### B) Deploy frontend to GitHub Pages

The `pages.yml` builds the frontend and deploys it to Pages. It also runs a backend import test with `SKIP_MODEL_LOAD=1` so the workflow doesn’t require model artifacts during the Pages build step.

---

# Summary

This project demonstrates the full lifecycle of a medical imaging application, starting from dataset preparation and CNN model training and progressing through inference validation, large-model artifact handling, deployment on free cloud infrastructure, containerization, frontend development, and automated CI/CD.

1. Dataset folder structure
2. Train model → save `artifacts/models/v1/*`
3. Evaluate → save report
4. FastAPI inference locally + curl tests
5. Dockerize FastAPI → test container locally
6. Push Docker image to Docker Hub
7. Upload large model to HF Hub (or Git LFS)
8. Deploy HF Space (Gradio) that downloads model at runtime
9. Test Gradio API programmatically
10. Build React frontend
11. Add CI/CD (CI + Pages deploy)

Github: [https://github.com/mdislammazharul/Lung_Cancer_Detection](https://github.com/mdislammazharul/Lung_Cancer_Detection)

Live Site: [https://mdislammazharul.github.io/Lung_Cancer_Detection/](https://mdislammazharul.github.io/Lung_Cancer_Detection/)


---
