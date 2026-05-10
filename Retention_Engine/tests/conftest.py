from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.churn_predictor import SklearnChurnPredictor
from app.services.customer_repository import CSVCustomerRepository


@pytest.fixture(scope="session")
def real_repository():
    """Real CSV repository — loaded once for the whole test session."""
    return CSVCustomerRepository(settings.customer_db_path)


@pytest.fixture(scope="session")
def real_predictor():
    """Real sklearn model — loaded once for the whole test session."""
    return SklearnChurnPredictor(settings.model_path)


@pytest.fixture
def client():
    """TestClient with full lifespan — model and CSV loaded, LLM is DemoLLMService."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_llm_service(client):
    """Returns the orchestrator's live LLM instance so tests can patch generate().

    Usage in tests:
        from unittest.mock import AsyncMock
        mock_llm_service.generate = AsyncMock(return_value=LLMResult(text=..., ...))
    """
    return client.app.state.orchestrator._llm


@pytest.fixture(scope="session")
def sample_customer_id(real_repository):
    """First customer ID from the database."""
    return real_repository.list_customer_ids()[0]


@pytest.fixture(scope="session")
def sample_record(real_repository, sample_customer_id):
    """Raw feature dict for the first customer."""
    return real_repository.get_customer(sample_customer_id)
