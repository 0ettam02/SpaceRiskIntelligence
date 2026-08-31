import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # Il context manager attiva la lifespan (caricamento dati + training dei
    # 5 modelli): una sola volta per l'intera sessione di test, non per ogni test.
    with TestClient(app) as test_client:
        yield test_client
