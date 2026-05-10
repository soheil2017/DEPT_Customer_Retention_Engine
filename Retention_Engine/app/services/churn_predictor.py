"""Churn probability predictor.

Wraps the serialised sklearn Pipeline so the rest of the application never
imports sklearn directly.  The Pipeline includes its own preprocessor
(ColumnTransformer with StandardScaler + OneHotEncoder), so we only need to
supply a raw DataFrame with the original feature columns.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.core.exceptions import ModelNotLoadedError
from app.services.interfaces import ChurnPredictorABC

logger = logging.getLogger(__name__)


class SklearnChurnPredictor(ChurnPredictorABC):
    """Loads a joblib-serialised sklearn Pipeline and runs predict_proba."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._pipeline = self._load_pipeline()

    #  Public interface 

    def predict(self, customer_features: dict[str, Any]) -> float:
        """Return churn probability ∈ [0, 1] for the given feature dict."""
        if self._pipeline is None:
            raise ModelNotLoadedError("Churn model is not available.")

        expected_cols = list(self._pipeline.feature_names_in_)
        row = pd.DataFrame([{col: customer_features.get(col) for col in expected_cols}])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probability: float = self._pipeline.predict_proba(row)[0][1]

        return round(float(probability), 4)

    #  Helpers 

    def _load_pipeline(self):
        logger.info("Loading churn model from %s", self._model_path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline = joblib.load(self._model_path)
            logger.info("Churn model loaded successfully.")
            return pipeline
        except Exception as exc:
            raise ModelNotLoadedError(f"Failed to load model: {exc}") from exc
