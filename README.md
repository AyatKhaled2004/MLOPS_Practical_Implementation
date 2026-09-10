# 🏠 Bangalore House Price Predictor — MLOps Practical Implementation

[![CI/CD Pipeline](https://github.com/AyatKhaled2004/MLOPS_Practical_Implementation/actions/workflows/main.yml/badge.svg)](https://github.com/AyatKhaled2004/MLOPS_Practical_Implementation/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/ayatkhaled/house-price-app)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An end-to-end **MLOps** project that trains a machine learning model to predict real‑estate prices in Bangalore, and serves it in production through a **FastAPI** backend and a **Streamlit** frontend — fully containerized, tested, tracked with **MLflow**, version‑controlled with **DVC**, and automatically built & deployed via a **GitHub Actions CI/CD pipeline**.

This repository is a practical, hands-on implementation of core MLOps concepts: data/model versioning, experiment tracking, automated testing, containerization, and continuous integration/continuous delivery.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Install dependencies](#2-install-dependencies)
  - [3. Pull the model artifacts (DVC)](#3-pull-the-model-artifacts-dvc)
  - [4. Train the model (optional)](#4-train-the-model-optional)
  - [5. Run the application](#5-run-the-application)
- [Running with Docker](#-running-with-docker)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Experiment Tracking with MLflow](#-experiment-tracking-with-mlflow)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Author](#-author)

---

## 🔎 Overview

The project solves a regression problem: **predicting the price of a house in Bangalore** based on its location, total square footage, number of bathrooms, and number of bedrooms (BHK). It's built around a clean separation of concerns that mirrors a real production ML system:

- **Model training** is a reproducible pipeline (`src/train.py`) that logs parameters, metrics, and the trained model to **MLflow**.
- **Model & data artifacts** (the trained model, dataset, and encoded column schema) are version-controlled with **DVC** instead of being committed to Git directly.
- **Inference** is exposed through a **FastAPI** REST endpoint (`src/main.py` → `/predict`), validated with **Pydantic** schemas.
- A **Streamlit** app (`app.py`) provides a simple, user-friendly UI that calls the API and displays the predicted price.
- **Structured logging** (`structlog`) captures every prediction request and failure in JSON format, ready for log aggregation tools.
- **Pytest** covers the prediction logic, and every push/PR is validated and containerized automatically by **GitHub Actions**.
- The whole application (API + UI) ships as a **single multi-stage Docker image**, pushed to Docker Hub on every merge to `main`.

## 🏗 Architecture

```
┌─────────────────┐        HTTP POST /predict        ┌──────────────────┐
│   Streamlit UI   │  ───────────────────────────────▶ │   FastAPI Server │
│    (app.py)      │ ◀─────────────────────────────── │   (src/main.py)  │
└─────────────────┘        JSON: predicted_price       └────────┬─────────┘
                                                                 │
                                                                 ▼
                                                     ┌───────────────────────┐
                                                     │  predict_price()       │
                                                     │  (src/predict.py)      │
                                                     │  loads model + columns │
                                                     └────────────┬───────────┘
                                                                  │
                                                                  ▼
                                                     ┌───────────────────────┐
                                                     │  artifacts/ (via DVC)  │
                                                     │  • model.pickle        │
                                                     │  • columns.json        │
                                                     │  • bhp.csv (raw data)  │
                                                     └───────────────────────┘

Training pipeline (src/train.py) ──▶ logs runs/metrics/model to MLflow ──▶ saves model.pickle to artifacts/
```

Both the FastAPI backend (port `8000`) and the Streamlit frontend (port `8501`) run **inside the same container**, started together via the Docker entrypoint.

## 🧰 Tech Stack

| Layer                  | Technology                                   |
|-------------------------|-----------------------------------------------|
| Model                   | scikit-learn (Linear Regression)              |
| Experiment Tracking     | MLflow                                        |
| Data/Model Versioning   | DVC                                           |
| Backend API             | FastAPI + Pydantic + Uvicorn                  |
| Frontend                | Streamlit                                     |
| Logging                 | structlog (structured JSON logs)              |
| Testing                 | Pytest                                        |
| Dependency Management   | uv                                             |
| Containerization        | Docker (multi-stage build)                    |
| CI/CD                   | GitHub Actions → Docker Hub                   |

## 📁 Project Structure

```
MLOPS_Practical_Implementation/
├── app.py                     # Streamlit frontend
├── Dockerfile                 # Multi-stage build (API + UI in one image)
├── pyproject.toml             # Project metadata & dependencies (uv)
├── artifacts.dvc              # DVC pointer to the artifacts/ directory
├── notebok/
│   └── house-prediction.ipynb # Exploratory data analysis & model prototyping
├── src/
│   ├── main.py                 # FastAPI app & /predict endpoint
│   ├── predict.py              # Model loading & inference logic
│   ├── train.py                # Training pipeline with MLflow tracking
│   ├── schemas.py               # Pydantic request/response models
│   ├── config.py                # Paths & configuration constants
│   └── logger.py                # structlog configuration
├── tests/
│   ├── conftest.py
│   └── test_predict.py         # Unit tests for prediction logic
└── .github/workflows/          # CI/CD pipeline (tests + Docker build & push)
```

> **Note:** `artifacts/` (containing `bhp.csv`, `columns.json`, and the trained `.pickle` model) is **not stored in Git** — it's tracked with DVC and must be pulled separately (see below).

## 🚀 Getting Started

### Prerequisites

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- [DVC](https://dvc.org/) — for pulling the dataset/model artifacts
- Docker (optional, for containerized run)

### 1. Clone the repository

```bash
git clone https://github.com/AyatKhaled2004/MLOPS_Practical_Implementation.git
cd MLOPS_Practical_Implementation
```

### 2. Install dependencies

```bash
pip install uv
uv sync
```

### 3. Pull the model artifacts (DVC)

The dataset, encoded columns file, and trained model live in `artifacts/` and are tracked by DVC rather than Git.

```bash
dvc pull
```

> If you don't have access to the configured DVC remote, you can instead run the training pipeline yourself (step 4) to regenerate the artifacts locally from `notebok/house-prediction.ipynb`'s exported data.

### 4. Train the model (optional)

Re-runs the training pipeline and logs a new run to MLflow:

```bash
uv run python src/train.py
```

To inspect experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 5. Run the application

Start the FastAPI backend:

```bash
uv run uvicorn src.main:app --reload --port 8000
```

In a separate terminal, start the Streamlit frontend:

```bash
uv run streamlit run app.py
```

- API docs (Swagger UI): **http://127.0.0.1:8000/docs**
- Streamlit app: **http://127.0.0.1:8501**

## 🐳 Running with Docker

The project ships a production-ready, multi-stage Dockerfile that runs **both** the API and the UI in a single container.

```bash
docker build -t house-price-app .
docker run -p 8000:8000 -p 8501:8501 house-price-app
```

Or pull the pre-built image published by the CI/CD pipeline:

```bash
docker run -p 8000:8000 -p 8501:8501 ayatkhaled/house-price-app:latest
```

## 📡 API Reference

### `GET /`

Health check — confirms the service is running.

```json
{ "status": "ok", "message": "Bangalore Real Estate API is running" }
```

### `POST /predict`

**Request body:**

```json
{
  "location": "1st Phase JP Nagar",
  "sqft": 1200,
  "bath": 2,
  "bhk": 3
}
```

**Response:**

```json
{ "predicted_price": 83.45 }
```

Example with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"location": "1st Phase JP Nagar", "sqft": 1200, "bath": 2, "bhk": 3}'
```

All inputs are validated with Pydantic (`sqft` between 100–50000, `bath`/`bhk` between 1–10).

## ✅ Testing

Unit tests cover the core prediction logic, including handling of unknown locations:

```bash
uv run pytest
```

## 🔄 CI/CD Pipeline

Every push and pull request to `main` triggers a **GitHub Actions** workflow (`.github/workflows/`) that:

1. Sets up Python via `uv` and installs dependencies (`uv sync --frozen`).
2. Runs the full **Pytest** suite.
3. On a successful push to `main`, builds the Docker image and pushes it to **Docker Hub** (`ayatkhaled/house-price-app:latest`).

This guarantees that only tested, working code ever gets containerized and shipped.

## 📊 Experiment Tracking with MLflow

Each training run logs:

- **Parameters:** model type, number of features
- **Metrics:** R² score, RMSE
- **Artifacts:** the trained scikit-learn model

Runs are stored locally in a SQLite backend (`mlflow.db`) and can be explored with `mlflow ui`.

## 🗺 Roadmap

- [ ] Add model versioning/registry via MLflow Model Registry
- [ ] Add more advanced models (e.g., XGBoost) with hyperparameter tuning
- [ ] Add monitoring/observability for the deployed API (e.g., Prometheus + Grafana)
- [ ] Deploy to a cloud platform (AWS/Azure/GCP) with a managed CI/CD flow
- [ ] Add authentication to the API

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and share it.

## 👤 Author

**Ayat Khaled**
📧 ayatkhaled32721@gmail.com
🔗 [GitHub](https://github.com/AyatKhaled2004)

---

⭐ If you found this project useful, consider giving it a star!