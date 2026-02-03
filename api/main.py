from typing import List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import uuid
import time
import logging

from api.services.preprocessing import (
    preprocess_request,
    preprocess_request_batch
)
from api.services.inference import (
    run_inference,
    run_inference_batch
)
from api.services.metrics import metrics_store

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("credit-risk-api")

# FastAPI App
app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    description="Production-ready Credit Risk Scoring Service",
)

# Middleware: Request ID + timing


@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    request.state.request_id = request_id

    response = await call_next(request)

    duration_ms = round((time.time() - start_time) * 1000, 2)

    response.headers["X-Request-ID"] = request_id

    metrics_store.record_request(duration_ms)

    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status_code={response.status_code} "
        f"duration_ms={duration_ms}"
    )

    return response

# Request schema


class CreditApplicationRequest(BaseModel):
    person_age: int = Field(..., example=35)
    person_income: float = Field(..., example=75000)
    person_home_ownership: str = Field(..., example="RENT")
    person_emp_length: int = Field(..., example=5)
    loan_intent: str = Field(..., example="PERSONAL")
    loan_grade: str = Field(..., example="B")
    loan_amnt: float = Field(..., example=15000)
    loan_int_rate: float = Field(..., example=12.5)
    loan_percent_income: float = Field(..., example=0.2)
    cb_person_default_on_file: str = Field(..., example="N")
    cb_person_cred_hist_length: int = Field(..., example=8)

# Health check


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}

# metrics


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    return metrics_store.snapshot()


# Single prediction
@app.post("/predict", tags=["Prediction"])
def predict_credit_risk(
    payload: CreditApplicationRequest,
    request: Request
):
    try:
        request_id = request.state.request_id

        features = preprocess_request(payload.dict())

        result = run_inference(
            features=features,
            request_id=request_id
        )

        metrics_store.record_single()

        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception:
        logger.exception(
            f"request_id={request.state.request_id} error=prediction_failed"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error"
        )

# Batch prediction (MAX 500)


@app.post("/predict/batch", tags=["Prediction"])
def predict_credit_risk_batch(
    payloads: List[CreditApplicationRequest],
    request: Request
):
    batch_size = len(payloads)

    if batch_size == 0:
        raise HTTPException(status_code=400, detail="Batch is empty")

    if batch_size > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum limit of 500 records"
        )

    try:
        request_id = request.state.request_id

        raw_records = [p.dict() for p in payloads]
        features = preprocess_request_batch(raw_records)

        results = run_inference_batch(
            features=features,
            request_id=request_id
        )

        metrics_store.record_batch(batch_size)

        return {
            "batch_size": batch_size,
            "results": results
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception:
        logger.exception(
            f"request_id={request.state.request_id} error=batch_prediction_failed"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal batch prediction error"
        )

# Prediction + Explainability (Docker-only SHAP)


@app.post("/predict/explain", tags=["Explainability"])
def predict_with_explanation(
    payload: CreditApplicationRequest,
    request: Request
):
    try:
        # Lazy import — prevents startup failure
        from api.services.explainability import explain_prediction

        request_id = request.state.request_id
        features = preprocess_request(payload.dict())

        result = run_inference(
            features=features,
            request_id=request_id
        )

        metrics_store.record_single()

        explanation = explain_prediction(features)

        return {
            **result,
            "explanations": explanation
        }

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Explainability is only available in Docker runtime"
        )

    except Exception:
        logger.exception(
            f"request_id={request.state.request_id} error=explainability_failed"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal explainability error"
        )
