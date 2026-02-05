# Credit Risk Scoring API

A **production-ready Machine Learning inference service** built with **FastAPI + Docker**, serving a trained **Gradient Boosting credit risk model** for real-time and batch loan application scoring.

This project demonstrates **end-to-end ML serving engineering**, not just model training.

It covers:

* Deterministic preprocessing
* Strict schema validation
* Model inference
* Business decisioning
* Structured logging
* Observability & metrics
* Automated tests
* Containerization
* Cloud deployment

---

## 🌐 Live Demo (Railway)

Swagger UI
👉 [https://credit-risk-scoring-api-production.up.railway.app/docs](https://credit-risk-scoring-api-production.up.railway.app/docs)

Health
👉 [https://credit-risk-scoring-api-production.up.railway.app/health](https://credit-risk-scoring-api-production.up.railway.app/health)

Metrics
👉 [https://credit-risk-scoring-api-production.up.railway.app/metrics](https://credit-risk-scoring-api-production.up.railway.app/metrics)

---

# 🚀 Features

## Prediction

* Single scoring → `/predict`
* Batch scoring → `/predict/batch` (≤ 500 records)
* Explainability with SHAP → `/predict/explain`

## Observability

* Request IDs
* Structured JSON logs
* Latency tracking
* Health checks
* Lightweight metrics endpoint

## Engineering

* Clean service architecture
* Strict schema enforcement
* Deterministic preprocessing
* Artifact-driven inference
* Pytest tests
* Fully Dockerized
* Cloud deployable

---

# 🧠 System Architecture

```
Client Request
      ↓
FastAPI Validation
      ↓
Preprocessing (cleaning + feature engineering)
      ↓
Schema Enforcement
      ↓
Model Inference
      ↓
Business Decision Logic
      ↓
Response + Logging + Metrics
```

### Key principle

**Model predicts risk.
Business layer makes decisions.**

This separation allows:

* policy changes without retraining
* safer production behavior
* clearer ownership

---

# 📦 Project Structure

```
credit_risk_api/
│
├── api/
│   ├── main.py                 # FastAPI entrypoint + middleware + routes
│   └── services/
│       ├── preprocessing.py
│       ├── inference.py
│       ├── explainability.py
│       └── metrics.py
│
├── pipeline/
│   ├── cleaning.py
│   └── features.py
│
├── artifacts/
│   └── model/
│       ├── gradient_boosting_model.joblib
│       ├── feature_schema.json
│       ├── decision_threshold.json
│       └── model_metadata.json
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🧪 API Endpoints

## Health

GET `/health`

```json
{ "status": "ok" }
```

## Version

GET `/version`

```json
{
  "service": "credit-risk-api",
  "model_version": "v1.0.0"
}
```

## Single Prediction

POST `/predict`

```json
{
  "decision": "APPROVE",
  "prediction": 0,
  "probability_of_default": 0.049
}
```

## Batch Prediction

POST `/predict/batch`

Vectorized scoring for multiple records.

## Metrics

GET `/metrics`

Returns:

* total requests
* latency
* decision counts
* batch vs single counts

---

# 📊 Structured Logging

All requests emit JSON logs:

```json
{
  "timestamp": "2026-02-04T12:09:51Z",
  "level": "INFO",
  "request_id": "abc123",
  "method": "POST",
  "path": "/predict",
  "duration_ms": 42.3
}
```

Benefits:

* tracing
* debugging
* monitoring
* production observability

---

# 🧪 Running Tests (Docker)

Tests run inside the **same runtime as production**.

```bash
docker build -t credit-risk-api .
docker run --rm credit-risk-api pytest -v
```

Example:

```
2 passed in 3.0s
```

---

# ⚙️ Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn api.main:app --reload
```

Swagger:

```
http://127.0.0.1:8000/docs
```

---

# 🐳 Docker

Build

```bash
docker build -t credit-risk-api .
```

Run

```bash
docker run -p 8000:8000 credit-risk-api
```

---

# 🐳 Docker Compose

```bash
docker compose up -d
```

Includes:

* API service
* health checks
* restart policy

---

# ☁️ Deployment Ready

Works with:

* Railway (current deployment)
* AWS ECS / Fargate
* Google Cloud Run
* Azure Container Apps
* Kubernetes

**Containerization ensures identical behavior everywhere.**

---

# 🧭 Project Goals

This project demonstrates:

* production ML serving
* clean API architecture
* artifact-driven inference
* observability practices
* containerized deployment
* testability
* cloud readiness

---

# 👤 Author

Built as an end-to-end demonstration of **real-world ML system engineering**, not just modeling.

Designed to reflect how production credit risk systems are deployed in fintech environments.
