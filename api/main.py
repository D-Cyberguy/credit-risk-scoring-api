from typing import List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import uuid
import time
import logging
import json

from api.services.preprocessing import (
    preprocess_request,
    preprocess_request_batch
)
from api.services.inference import (
    run_inference,
    run_inference_batch
)
from api.services.metrics import metrics_store


# JSON logging configuration
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())

logger = logging.getLogger("credit-risk-api")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(handler)


# App
app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    description="Production-ready Credit Risk Scoring Service",
)


# Middleware: request id + latency
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
    person_age: int = Field(..., json_schema_extra={"example": 35})
    person_income: float = Field(..., json_schema_extra={"example": 75000})
    person_home_ownership: str = Field(..., json_schema_extra={
                                       "example": "RENT"})
    person_emp_length: int = Field(..., json_schema_extra={"example": 5})
    loan_intent: str = Field(..., json_schema_extra={"example": "PERSONAL"})
    loan_grade: str = Field(..., json_schema_extra={"example": "B"})
    loan_amnt: float = Field(..., json_schema_extra={"example": 15000})
    loan_int_rate: float = Field(..., json_schema_extra={"example": 12.5})
    loan_percent_income: float = Field(..., json_schema_extra={"example": 0.2})
    cb_person_default_on_file: str = Field(..., json_schema_extra={
                                           "example": "N"})
    cb_person_cred_hist_length: int = Field(..., json_schema_extra={
                                            "example": 8})


# Health
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}


# Version
@app.get("/version", tags=["System"])
def version():
    return {
        "service": "credit-risk-api",
        "model_name": "credit_risk_gradient_boosting",
        "model_version": "v1.0.0"
    }


# Metrics
@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    return metrics_store.snapshot()


# Single prediction
@app.post("/predict", tags=["Prediction"])
def predict_credit_risk(payload: CreditApplicationRequest, request: Request):
    try:
        request_id = request.state.request_id

        features = preprocess_request(payload.model_dump())

        result = run_inference(
            features=features,
            request_id=request_id
        )

        metrics_store.record_single()

        logger.info(
            f"request_id={request_id} "
            f"prob={result['probability_of_default']} "
            f"decision={result['decision']}"
        )

        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception:
        logger.exception(
            f"request_id={request.state.request_id} error=prediction_failed"
        )
        raise HTTPException(500, "Internal prediction error")


# Batch prediction
@app.post("/predict/batch", tags=["Prediction"])
def predict_credit_risk_batch(payloads: List[CreditApplicationRequest], request: Request):
    batch_size = len(payloads)

    if batch_size == 0:
        raise HTTPException(400, "Batch is empty")

    if batch_size > 500:
        raise HTTPException(
            400, "Batch size exceeds maximum limit of 500 records")

    try:
        request_id = request.state.request_id

        raw_records = [p.model_dump() for p in payloads]
        features = preprocess_request_batch(raw_records)

        results = run_inference_batch(
            features=features,
            request_id=request_id
        )

        metrics_store.record_batch(batch_size)

        for r in results:
            logger.info(
                f"request_id={request_id} "
                f"prob={r['probability_of_default']} "
                f"decision={r['decision']}"
            )

        return {
            "batch_size": batch_size,
            "results": results
        }

    except ValueError as ve:
        raise HTTPException(400, str(ve))

    except Exception:
        logger.exception(
            f"request_id={request.state.request_id} error=batch_prediction_failed"
        )
        raise HTTPException(500, "Internal batch prediction error")


# Prediction + explainability
@app.post("/predict/explain", tags=["Explainability"])
def predict_with_explanation(payload: CreditApplicationRequest, request: Request):
    try:
        from api.services.explainability import explain_prediction

        request_id = request.state.request_id

        features = preprocess_request(payload.model_dump())

        result = run_inference(
            features=features,
            request_id=request_id
        )

        metrics_store.record_single()

        logger.info(
            f"request_id={request_id} "
            f"prob={result['probability_of_default']} "
            f"decision={result['decision']} "
            f"explainability=True"
        )

        explanation = explain_prediction(features)

        return {
            **result,
            "explanations": explanation
        }

    except ImportError:
        raise HTTPException(
            501,
            "Explainability is only available in Docker runtime"
        )

    except Exception:
        logger.exception(
            f"request_id={request.state.request_id} error=explainability_failed"
        )
        raise HTTPException(500, "Internal explainability error")
