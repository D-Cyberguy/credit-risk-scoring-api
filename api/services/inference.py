import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from api.services.metrics import metrics_store

logger = logging.getLogger("credit-risk-api")

# Resolve artifact paths
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "model"

MODEL_PATH = ARTIFACTS_DIR / "gradient_boosting_model.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "decision_threshold.json"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

# Load artifacts once
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    _thresholds = json.load(f)

with open(METADATA_PATH, "r") as f:
    _metadata = json.load(f)

APPROVE_THRESHOLD = _thresholds.get("approve", 0.3)
CONDITIONAL_THRESHOLD = _thresholds.get("conditional", 0.6)

MODEL_NAME = _metadata.get("model_name")
MODEL_VERSION = _metadata.get("model_version")


def _make_decision(probability: Optional[float]) -> str:
    if probability is None:
        return "UNKNOWN"
    if probability < APPROVE_THRESHOLD:
        return "APPROVE"
    if probability < CONDITIONAL_THRESHOLD:
        return "CONDITIONAL_APPROVAL"
    return "REJECT"


def run_inference(
    features: pd.DataFrame,
    request_id: Optional[str] = None
) -> Dict:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    prediction = int(model.predict(features)[0])

    probability = (
        float(model.predict_proba(features)[0][1])
        if hasattr(model, "predict_proba")
        else None
    )

    decision = _make_decision(probability)

    logger.info(
        f"request_id={request_id} "
        f"prediction={prediction} "
        f"pd={round(probability, 4) if probability is not None else 'NA'} "
        f"decision={decision} "
        f"model={MODEL_NAME} "
        f"version={MODEL_VERSION}"
    )

    metrics_store.record_decision(decision)

    return {
        "decision": decision,
        "prediction": prediction,
        "probability_of_default": probability,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


def run_inference_batch(
    features: pd.DataFrame,
    request_id: Optional[str] = None
) -> List[Dict]:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    predictions = model.predict(features)

    probabilities = (
        model.predict_proba(features)[:, 1]
        if hasattr(model, "predict_proba")
        else [None] * len(predictions)
    )

    results: List[Dict] = []

    for pred, prob in zip(predictions, probabilities):
        decision = _make_decision(prob)

        metrics_store.record_decision(decision)

        results.append({
            "decision": decision,
            "prediction": int(pred),
            "probability_of_default": float(prob) if prob is not None else None,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
        })

    logger.info(
        f"request_id={request_id} "
        f"batch_size={len(results)} "
        f"model={MODEL_NAME} "
        f"version={MODEL_VERSION}"
    )

    return results
