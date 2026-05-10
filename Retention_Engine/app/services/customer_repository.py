"""CSV-backed customer repository.

Loads the customer database once at startup and serves O(1) lookups via an
in-memory dict. This is a simple implementation for demonstration purposes; in production, you'd
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.exceptions import CustomerNotFoundError
from app.services.interfaces import CustomerRepositoryABC

logger = logging.getLogger(__name__)


_INACTIVE_VALUES = {"No", "No internet service", "No phone service", ""}


class CSVCustomerRepository(CustomerRepositoryABC):
    """Loads a CSV file into an in-memory lookup on construction."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    #  Public interface 
    def get_customer(self, customer_id: str) -> dict[str, Any]:
        try:
            return self._records[customer_id]
        except KeyError:
            raise CustomerNotFoundError(customer_id)

    def list_customer_ids(self) -> list[str]:
        return list(self._records.keys())

    #  Helpers 

    def _load(self) -> None:
        logger.info("Loading customer database from %s", self._db_path)
        df = pd.read_csv(self._db_path)

        if "customerID" not in df.columns:
            raise ValueError("Customer database CSV must contain a 'customerID' column.")

        df = df.set_index("customerID")
        self._records = df.to_dict(orient="index")
        logger.info("Loaded %d customer records.", len(self._records))

    @staticmethod
    def get_active_services(record: dict[str, Any]) -> list[str]:
        """Return a human-readable list of services the customer has active."""
        service_columns = [
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        active = []
        for col in service_columns:
            val = str(record.get(col, ""))
            if val not in _INACTIVE_VALUES:
                active.append(f"{col.replace('Service', ' Service')}: {val}")
        return active
