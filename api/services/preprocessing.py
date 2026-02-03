import json
from pathlib import Path
import pandas as pd

from pipeline.cleaning import clean_raw_data
from pipeline.features import engineer_features
from api.services.schema import validate_raw_schema

# Load feature schema (authoritative ML contract)
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "model"

FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"

with open(FEATURE_SCHEMA_PATH, "r") as f:
    _schema = json.load(f)

EXPECTED_FEATURES: list[str] = _schema["feature_names"]
NUM_FEATURES: int = _schema["num_features"]


# Internal validation
def _validate_feature_schema(df_features: pd.DataFrame) -> None:
    """
    Ensure engineered features exactly match the model contract.
    """
    missing = [c for c in EXPECTED_FEATURES if c not in df_features.columns]
    extra = [c for c in df_features.columns if c not in EXPECTED_FEATURES]

    if missing:
        raise ValueError(f"Missing required engineered features: {missing}")

    if extra:
        raise ValueError(f"Unexpected engineered features produced: {extra}")

    if df_features.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Feature count mismatch: expected {NUM_FEATURES}, "
            f"got {df_features.shape[1]}"
        )


# Public API — single request preprocessing
def preprocess_request(payload: dict) -> pd.DataFrame:
    """
    Preprocess a single credit application payload into model-ready features.
    """

    if not isinstance(payload, dict):
        raise TypeError("Payload must be a dictionary")

    # Convert to DataFrame
    df_raw = pd.DataFrame([payload])

    # 1️⃣ RAW schema validation (serving-time contract)
    validate_raw_schema(df_raw)

    # 2️⃣ Cleaning
    df_clean = clean_raw_data(df_raw)

    # 3️⃣ Feature engineering
    df_features = engineer_features(
        df_clean=df_clean,
        expected_features=EXPECTED_FEATURES,
    )

    # 4️⃣ Feature schema enforcement
    _validate_feature_schema(df_features)

    return df_features


# Public API — batch request preprocessing
def preprocess_request_batch(payloads: list[dict]) -> pd.DataFrame:
    """
    Preprocess a batch of credit application payloads into model-ready features.
    """

    if not isinstance(payloads, list):
        raise TypeError("Batch payload must be a list of dictionaries")

    if not payloads:
        raise ValueError("Batch payload is empty")

    # Convert batch to DataFrame
    df_raw = pd.DataFrame(payloads)

    # 1️⃣ RAW schema validation (once per batch)
    validate_raw_schema(df_raw)

    # 2️⃣ Cleaning (vectorized)
    df_clean = clean_raw_data(df_raw)

    # 3️⃣ Feature engineering (vectorized)
    df_features = engineer_features(
        df_clean=df_clean,
        expected_features=EXPECTED_FEATURES,
    )

    # 4️⃣ Feature schema enforcement
    _validate_feature_schema(df_features)

    return df_features
