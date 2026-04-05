import os

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_azure_env(monkeypatch):
    keys = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_PDF_API_KEY",
        "AZURE_OPENAI_PDF_ENDPOINT",
        "AZURE_OPENAI_PDF_DEPLOYMENT",
        "AZURE_OPENAI_PDF_API_VERSION",
        "AZURE_ANTHROPIC_API_KEY",
        "AZURE_ANTHROPIC_ENDPOINT",
        "AZURE_ANTHROPIC_MODEL",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield
