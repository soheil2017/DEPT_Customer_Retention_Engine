"""Tests for CSVCustomerRepository."""
import pytest

from app.core.exceptions import CustomerNotFoundError
from app.services.customer_repository import CSVCustomerRepository


def test_list_customer_ids_returns_non_empty_list(real_repository):
    ids = real_repository.list_customer_ids()
    assert len(ids) > 0


def test_get_customer_returns_feature_dict(real_repository, sample_customer_id):
    record = real_repository.get_customer(sample_customer_id)
    assert isinstance(record, dict)
    # All 18 model features must be present
    expected_keys = {
        "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
    }
    assert expected_keys.issubset(record.keys())


def test_get_customer_raises_for_unknown_id(real_repository):
    with pytest.raises(CustomerNotFoundError):
        real_repository.get_customer("DOES-NOT-EXIST")


def test_get_active_services_filters_inactive(real_repository, sample_record):
    services = CSVCustomerRepository.get_active_services(sample_record)
    for s in services:
        assert "No" not in s or ":" in s  # only "No" values that have an active type label
