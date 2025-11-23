---
title: "🫀 Building a Full-Stack Heart Disease Prediction System: ML, FastAPI, Docker, React, Render, GitHub Pages, and CI/CD"
date: 2025-11-23
permalink: /posts/2025-11-23-Full-Stack_Heart_Disease_Prediction_System.md/
tags:
  - ML
  - FastAPI
  - Docker
  - React
  - CI/CD
---

In this project, I built a **complete end-to-end AI application** that predicts the risk of **heart failure** using real clinical data. The goal was not just to train a machine learning model, but to **transform it into a full-stack production-grade web application**, deploy it using modern DevOps tools, and make it publicly available.

This system includes:

| Layer              | Technology                               |
| ------------------ | ---------------------------------------- |
| ML Training        | Python, Scikit-Learn, Matplotlib, Pandas |
| Backend API        | FastAPI + Uvicorn                        |
| Model Serving      | Pickle + Docker                          |
| Deployment (API)   | Render (Docker)                          |
| Frontend           | React + Vite + Tailwind                  |
| Hosting (Frontend) | GitHub Pages                             |
| CI/CD              | GitHub Actions                           |
| Communication      | REST API (JSON)                          |

# 🧠 System Architecture

```
                        ┌────────────────────────────┐
                        │    Machine Learning (ML)   │
                        │  Model training + Pickle   │
                        │  Python, Scikit-learn      │
                        └─────────────┬──────────────┘
                                      │
                              Model.pkl exported
                                      │
                        ┌─────────────▼──────────────┐
                        │  FastAPI Backend (Render)  │
                        │ /predict endpoint (JSON)   │
                        │ Dockerized model serving   │
                        └─────────────┬──────────────┘
                                      │
                                  REST API
                                      │
                     ┌────────────────▼────────────────┐
                     │        React Frontend           │
                     │   Hosted on GitHub Pages        │
                     │   Form → API → Prediction       │
                     └────────────────┬────────────────┘
                                      │
                                      │
                        ┌─────────────▼──────────────┐
                        │    End Users (Browser)     │
                        │ Web Interface for Testing  │
                        └────────────────────────────┘
```

---

# 📂 File Structure

```
Heart_Disease/
│── .python-version
│── Dockerfile
│── pyproject.toml / requirements.txt
│── uv.lock
│
│── heart_failure_clinical_records_dataset.csv
│── heart_failure_model.pkl
│── Mid_Term_Project.py       # Training Script
│── Mid_Term_Project.ipynb    # EDA Notebook
│── export_artifacts.py       # JSON/Graph exporting
│── main.py                   # FastAPI backend
│
├── figures/
│   ├── correlation_matrix.png
│   ├── histograms.png
│   ├── death_event.png
│
├── heart-disease-app/        # FRONTEND (React+Vite)
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DataHead.jsx
│   │   │   ├── EDAGallery.jsx
│   │   │   ├── ModelSummary.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│
└── .github/
    └── workflows/
        └── deploy.yml        # CI/CD GitHub Actions
```

---

# ⚙️ Phase 1: Machine Learning Model Development

### 📥 Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 🧪 Load Dataset & Explore

```python
import pandas as pd
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
print(df.head())
df.info()
df.describe()
```

### 📊 EDA Visualizations

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=True)
plt.savefig('figures/correlation_matrix.png')
```

### 🤖 Model Training

```bash
python Mid_Term_Project.py
```

Inside `Mid_Term_Project.py`:

- Model training
- Cross-validation
- Hyperparameter tuning
- Save best model:

```python
import pickle
pickle.dump(best_model, open("heart_failure_model.pkl", "wb"))
```

---

# 🌐 Phase 2: FastAPI Backend

### 📦 Install FastAPI

```bash
pip install fastapi uvicorn pydantic gunicorn
```

### 🚀 Create API (main.py)

```python
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()
model = pickle.load(open("heart_failure_model.pkl", "rb"))

@app.post("/predict")
def predict(request: dict):
    values = np.array([list(request.values())]).reshape(1,-1)
    prediction = model.predict(values)[0]
    probability = model.predict_proba(values)[0][1]
    return {"prediction": int(prediction),
            "probability": float(probability)}
```

### ▶️ Run Locally

```bash
uvicorn main:app --reload --port 8000
```

Visit: `http://127.0.0.1:8000/docs`

---

# 🐳 Phase 3: Dockerization

### 📝 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--log-file", "-"]
```

### 🏗️ Build Docker Image

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

### 📤 Push to Docker Hub

```bash
docker tag heart-disease-api mdislammazharul/heart-disease-api
docker push mdislammazharul/heart-disease-api
```

---

# 🚀 Phase 4: Deploy Backend using Render (Docker)

### Steps:

1️⃣ Render site → [https://dashboard.render.com](https://dashboard.render.com)

2️⃣ New → **Web Service**

3️⃣ Select **Deploy from Docker**

4️⃣ Use GitHub repo

5️⃣ Use start command auto-detected from Dockerfile.

6️⃣ Deploy — after build, get an API URL like:

```
https://heart-failure-prediction-qe7o.onrender.com/predict
```

---

# 🌐 Phase 5: React Frontend (Vite + Tailwind)

### 💻 Create Project

```bash
npm create vite@latest heart-disease-app --template react
cd heart-disease-app
npm install
```

### 🎨 Install Tailwind

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### 🌐 Call API from Frontend

In `src/components/PredictForm.jsx`:

```javascript
const API_BASE = "https://heart-failure-prediction-qe7o.onrender.com/predict";

fetch(API_BASE, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(formData),
})
  .then((res) => res.json())
  .then((data) => setResult(data));
```

---

# ⚠️ Configure CORS on FastAPI (main.py)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://mdislammazharul.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

# 🚀 Deploy Frontend using GitHub Pages

```bash
npm install gh-pages --save-dev
```

In `package.json`:

```json
"homepage": "https://mdislammazharul.github.io/Heart_Failure_Prediction/",
"scripts": {
  "deploy": "gh-pages -d dist"
}
```

Build and deploy:

```bash
npm run build
npm run deploy
```

Frontend is live at:

👉 `https://mdislammazharul.github.io/Heart_Failure_Prediction/`

---

# 🔄 CI/CD with GitHub Actions

Create: `.github/workflows/deploy.yml`

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: ["main"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm install
      - run: npm run build
      - uses: JamesIves/github-pages-deploy-action@v4
        with:
          folder: dist
```

---

# 🎯 Live Demo

🔗 Frontend: **[https://mdislammazharul.github.io/Heart_Failure_Prediction/](https://mdislammazharul.github.io/Heart_Failure_Prediction/)**

🔗 API Endpoint: **[https://heart-failure-prediction-qe7o.onrender.com/docs](https://heart-failure-prediction-qe7o.onrender.com/docs)**

---
