Credit Risk Scoring API
A production-ready Machine Learning inference service built with FastAPI + Docker, serving a trained Gradient Boosting credit risk model for real-time and batch loan application scoring.
This project demonstrates end-to-end ML serving engineering, not just modeling.

It includes:


strict schema validation


deterministic preprocessing


model inference


business decision logic


request tracing


structured logging


lightweight metrics


Dockerized deployment


automated tests



🚀 Features
Prediction


Single scoring (/predict)


Batch scoring (/predict/batch, up to 500)


Explainability with SHAP (/predict/explain)


Observability


Request IDs


Structured JSON logs


Latency tracking


In-memory metrics endpoint


Health checks


Engineering


Clean service architecture


Model artifact separation


Pytest tests


Dockerized runtime


Cloud deployment ready



🧠 Architecture
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

Key principle:

Model predicts risk
Business layer makes decisions


📦 Project Structure
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


🧪 API Endpoints
Health
GET /health

{ "status": "ok" }


Version
GET /version

{
  "service": "credit-risk-api",
  "model_version": "v1.0.0"
}


Single Prediction
POST /predict

Returns:
{
  "decision": "APPROVE",
  "prediction": 0,
  "probability_of_default": 0.049
}


Batch Prediction
POST /predict/batch

Vectorized scoring for multiple records.

Metrics
GET /metrics

Returns:


total requests


latency


decision counts


batch vs single counts




📊 Logging (Structured)
All requests generate JSON logs:
Example:
{
  "timestamp": "2026-02-04T12:09:51Z",
  "level": "INFO",
  "message": "request_id=abc123 method=POST path=/predict duration_ms=42.3"
}

Each prediction logs:
request_id, probability, decision

Useful for:


monitoring


debugging


tracing


production observability




🧪 Running Tests (Docker)
We test inside the same runtime as production:
docker build -t credit-risk-api .
docker run --rm credit-risk-api pytest -v

Example:
2 passed in 3.0s



⚙️ Run Locally (No Docker)
python -m venv .venv # python version depends on you
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload

Docs:
http://127.0.0.1:8000/docs



🐳 Run With Docker
Build:
docker build -t credit-risk-api .

Run:
docker run -p 8000:8000 credit-risk-api



🐳 Docker Compose
docker compose up -d

Includes:


API


healthcheck


restart policy




☁️ Deployment Ready
Designed for:


AWS ECS / Fargate


Google Cloud Run


Azure Container Apps


Kubernetes


Containerized = same behavior everywhere.


WHICH EVER OF THE ABOVE WORKS FOR YOU!