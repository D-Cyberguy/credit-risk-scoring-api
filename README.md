Credit Risk Scoring API

A production-oriented Credit Risk Scoring API built with FastAPI, serving a trained Gradient Boosting model for real-time and batch loan application scoring.

This project focuses on end-to-end ML serving design, including preprocessing, schema enforcement, inference, business decisioning, lightweight monitoring, and containerized deployment.

⚠️ Project status: Actively evolving. Additional enhancements and refinements will be added over time.

🚀 What This Project Does

Scores loan applications for probability of default

Supports:

Real-time scoring for single applications

Batch scoring (up to 500 applications per request)

Cleanly separates:

Model prediction (binary default / no default)

Business decision logic (APPROVE / CONDITIONAL_APPROVAL / REJECT)

Enforces:

Raw input validation

Deterministic preprocessing

Strict alignment with the model’s trained feature schema

Provides:

Request-level logging with request IDs

Lightweight operational metrics via /metrics

Fully containerized with Docker, ready for deployment

🧠 High-Level Architecture
Client Request
   ↓
FastAPI Request Validation
   ↓
Raw Schema Validation (categorical & semantic checks)
   ↓
Data Cleaning (type coercion, clipping, missing handling)
   ↓
Feature Engineering (one-hot encoding, ordinal mapping)
   ↓
Feature Schema Enforcement (artifact-driven)
   ↓
Model Inference (Gradient Boosting)
   ↓
Business Decision Layer
   ↓
JSON Response


Key idea:

The model predicts risk.
The system applies policy to make decisions.

📦 Project Structure
credit_risk_api/
│
├── api/
│   ├── main.py                  # FastAPI entry point, middleware, routes
│   └── services/
│       ├── schema.py            # Raw input schema validation
│       ├── preprocessing.py     # Validation → cleaning → feature engineering
│       ├── inference.py         # Model loading, inference & decision logic
│       └── metrics.py           # Lightweight in-memory metrics store
│
├── pipeline/
│   ├── cleaning.py              # Deterministic data cleaning logic
│   └── features.py              # Feature engineering logic
│
├── artifacts/
│   └── model/
│       ├── gradient_boosting_model.joblib
│       ├── feature_schema.json
│       ├── decision_threshold.json
│       └── model_metadata.json
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

🔍 Model & Artifacts

Model: GradientBoostingClassifier (scikit-learn)

Objective: Recall-prioritized default detection

Training performance: ROC-AUC ≈ 0.95

Artifacts are externalized to support:

Reproducibility

Versioning

Safe inference

Clear separation between training and serving

No preprocessing or training logic is executed at runtime.

🧪 API Endpoints
Health Check

GET /health

{ "status": "ok" }

Single Prediction

POST /predict

Example request:

{
  "person_age": 32,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 4,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 12000,
  "loan_int_rate": 13.5,
  "loan_percent_income": 0.25,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}


Example response:

{
  "decision": "APPROVE",
  "prediction": 0,
  "probability_of_default": 0.0493,
  "model_name": "credit_risk_gradient_boosting",
  "model_version": "v1.0.0"
}

Batch Prediction

POST /predict/batch

Accepts a list of applications

Vectorized preprocessing & inference

Maximum batch size: 500

Example response:

{
  "batch_size": 2,
  "results": [
    { "decision": "APPROVE", "prediction": 0, "probability_of_default": 0.04 },
    { "decision": "REJECT", "prediction": 1, "probability_of_default": 0.82 }
  ]
}

📊 Monitoring & Metrics
Metrics Endpoint

GET /metrics

Provides lightweight operational statistics, including:

Total request count

Average request latency

Decision distribution (APPROVE / CONDITIONAL / REJECT)

Batch vs single request counts

Metrics are stored in-memory for simplicity and clarity.
This design demonstrates monitoring concepts without introducing infrastructure complexity.

🧮 Model Output vs Business Decision

Model output:

Binary prediction (default / no default)

Probability of default

Business decision:
Derived from configurable probability thresholds:

Probability of Default	Decision
Low	APPROVE
Medium	CONDITIONAL_APPROVAL
High	REJECT

Thresholds are externalized in decision_threshold.json, allowing policy changes without retraining the model.

⚙️ Running Locally (Without Docker)
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn api.main:app --reload


Swagger UI: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

🐳 Docker & Containerized Execution
Build Image
docker build -t credit-risk-api:latest .

Run Container
docker run -d \
  --name credit-risk-api \
  -p 8000:8000 \
  credit-risk-api:latest


Verify:

curl http://localhost:8000/health

🧩 Docker Compose

A minimal docker-compose.yml is included to support future service expansion (e.g., databases, message queues, monitoring).

docker compose up -d
docker compose down


The current compose setup runs the API only. Additional services can be added incrementally without changing application code.

🏗️ Deployment Readiness

This project is designed to be deployable on:

AWS ECS / Fargate

Google Cloud Run

Azure Container Apps

Kubernetes (with minimal adjustments)

Containerization ensures:

Environment consistency

Predictable runtime behavior

Smooth CI/CD integration

🧭 Project Status

This project is actively evolving.
Future improvements may include additional validation, explainability, enhanced monitoring, or deployment-related enhancements.

👤 Author Notes

This project was built to demonstrate:

End-to-end ML serving architecture

Clean separation of concerns

Artifact-driven inference

Practical credit risk system design

Production-minded FastAPI engineering