"""Tests for SklearnChurnPredictor."""
import pytest

from app.core.exceptions import ModelNotLoadedError


def test_predict_returns_probability_in_range(real_predictor, sample_record):
    prob = real_predictor.predict(sample_record)
    assert 0.0 <= prob <= 1.0


def test_predict_returns_float(real_predictor, sample_record):
    prob = real_predictor.predict(sample_record)
    assert isinstance(prob, float)


def test_predict_all_customers_in_valid_range(real_repository, real_predictor):
    for cid in real_repository.list_customer_ids():
        record = real_repository.get_customer(cid)
        prob = real_predictor.predict(record)
        assert 0.0 <= prob <= 1.0, f"Out-of-range probability for {cid}: {prob}"
