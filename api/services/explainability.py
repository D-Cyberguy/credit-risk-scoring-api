import pandas as pd
from typing import Dict, List
import hashlib
import json

from api.services.inference import model  # use already-loaded model


# In-memory cache (process-level)
_SHAP_CACHE: Dict[str, Dict] = {}

# Lazy globals
_explainer = None


def _get_explainer():
    global _explainer

    if _explainer is None:
        try:
            import shap
        except ImportError as e:
            raise RuntimeError(
                "SHAP is not installed. Explainability is only available in Docker."
            ) from e

        _explainer = shap.TreeExplainer(model)

    return _explainer


def _hash_features(features: pd.DataFrame) -> str:
    payload = features.iloc[0].to_dict()
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def explain_prediction(
    features: pd.DataFrame,
    top_k: int = 5
) -> Dict[str, List[Dict]]:

    if features.shape[0] != 1:
        raise ValueError("Explainability only supports single-record input")

    cache_key = _hash_features(features)

    # Cache hit
    if cache_key in _SHAP_CACHE:
        return _SHAP_CACHE[cache_key]

    explainer = _get_explainer()
    shap_values = explainer.shap_values(features)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    contributions = (
        pd.DataFrame({
            "feature": features.columns,
            "impact": shap_values[0]
        })
        .sort_values("impact", ascending=False)
    )

    explanation = {
        "risk_drivers": [
            {"feature": r.feature, "impact": round(r.impact, 4)}
            for r in contributions.head(top_k).itertuples()
        ],
        "protective_factors": [
            {"feature": r.feature, "impact": round(r.impact, 4)}
            for r in contributions.tail(top_k).sort_values("impact").itertuples()
        ]
    }

    _SHAP_CACHE[cache_key] = explanation
    return explanation
