import types

import pytest

import app.services.embedding_service as embedding_service_module
from app.services.embedding_service import EmbeddingService


class _FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddingsClient:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, model, input):
        if self._error:
            raise self._error
        return self._response


class _FakeAzureClient:
    def __init__(self, response=None, error=None):
        self.embeddings = _FakeEmbeddingsClient(response=response, error=error)


def test_embedding_service_not_configured_raises_runtime_error(monkeypatch):
    service = EmbeddingService()

    with pytest.raises(RuntimeError) as exc:
        service.generate_embedding("hello")

    assert "Azure OpenAI not configured" in str(exc.value)


def test_embedding_service_success(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")

    fake_response = _FakeResponse([_FakeEmbeddingItem([0.1, 0.2, 0.3])])

    def _fake_client(**kwargs):
        return _FakeAzureClient(response=fake_response)

    monkeypatch.setattr(embedding_service_module, "AzureOpenAI", _fake_client)

    service = EmbeddingService()
    result = service.generate_embedding("text")

    assert result == [0.1, 0.2, 0.3]


def test_embedding_service_empty_response(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")

    def _fake_client(**kwargs):
        return _FakeAzureClient(response=_FakeResponse([]))

    monkeypatch.setattr(embedding_service_module, "AzureOpenAI", _fake_client)

    service = EmbeddingService()
    with pytest.raises(RuntimeError) as exc:
        service.generate_embedding("text")

    assert "Failed to generate embedding with Azure" in str(exc.value)


def test_embedding_service_client_exception(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")

    def _fake_client(**kwargs):
        return _FakeAzureClient(error=ValueError("boom"))

    monkeypatch.setattr(embedding_service_module, "AzureOpenAI", _fake_client)

    service = EmbeddingService()
    with pytest.raises(RuntimeError) as exc:
        service.generate_embedding("text")

    assert "Failed to generate embedding with Azure" in str(exc.value)
