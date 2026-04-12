import app.services.azure_llm_service as llm_module
from app.models.schemas import CandidateProfileInput
from app.services.azure_llm_service import AzureLLMService


class _FakeChoiceMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeChoiceMessage(content)


class _FakeCompletion:
    def __init__(self, choices):
        self.choices = choices


class _FakeChatCompletions:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


class _FakeChatClient:
    def __init__(self, response=None, error=None):
        self.completions = _FakeChatCompletions(response=response, error=error)


class _FakeAzureClient:
    def __init__(self, response=None, error=None):
        self.chat = _FakeChatClient(response=response, error=error)


def test_llm_service_not_configured_returns_none():
    service = AzureLLMService()

    assert service.is_configured() is False
    assert service.generate_match_feedback("c", "v", 80.0) is None
    assert service.evaluate_match_dimensions({}, {}) is None
    assert service.extract_candidate_profile("cv text") is None


def test_llm_service_match_feedback_success(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

    response = _FakeCompletion([_FakeChoice("Buen match")])

    def _fake_client(**kwargs):
        return _FakeAzureClient(response=response)

    monkeypatch.setattr(llm_module, "AzureOpenAI", _fake_client)

    service = AzureLLMService()
    result = service.generate_match_feedback("cand", "vac", 92.5)

    assert result == "Buen match"


def test_llm_service_extract_profile_success(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

    response = _FakeCompletion([
        _FakeChoice(
            '{"professionalTitle":"Backend Engineer","summary":"Summary","skills":["Python"],"experience":[{"role":"Backend Developer","project":"Order Management System","tech":["Java","Spring Boot"],"description":"APIs REST","start":"2023","end":"2024"}],"education":[{"institution":"ECI","degree":"Ingenieria de Sistemas","start":"2022","end":"2026"}],"location":"Bogota","languages":["es"],"expectedSalary":7000000,"availability":null,"email":"persona@email.com","phoneNumber":"+57 300 000 0000","github":"https://github.com/usuario","linkedin":"https://linkedin.com/in/usuario"}'
        )
    ])

    def _fake_client(**kwargs):
        return _FakeAzureClient(response=response)

    monkeypatch.setattr(llm_module, "AzureOpenAI", _fake_client)

    service = AzureLLMService()
    profile = service.extract_candidate_profile("cv text")

    assert isinstance(profile, CandidateProfileInput)
    assert profile.professional_title == "Backend Engineer"
    assert profile.expected_salary == 7000000
    assert profile.email == "persona@email.com"
    assert profile.experience[0].role == "Backend Developer"
    assert profile.education[0].institution == "ECI"


def test_llm_service_extract_profile_normalizes_sector(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

    response = _FakeCompletion([
        _FakeChoice(
            '{"professionalTitle":"Backend Engineer","summary":"Summary","skills":["Python"],"experience":[],"education":[],"location":"Bogota","nationality":"Colombia","languages":["Espanol"],"sector":"tech","expectedSalary":7000000,"availability":null,"email":"persona@email.com","phoneNumber":"+57 300 000 0000","github":[],"linkedin":[],"links":[]}'
        )
    ])

    def _fake_client(**kwargs):
        return _FakeAzureClient(response=response)

    monkeypatch.setattr(llm_module, "AzureOpenAI", _fake_client)

    service = AzureLLMService()
    profile = service.extract_candidate_profile("cv text")

    assert isinstance(profile, CandidateProfileInput)
    assert profile.sector == "Tecnologia"
    assert profile.nationality == "Colombia"


def test_llm_service_evaluate_match_dimensions_success(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

    response = _FakeCompletion([
        _FakeChoice(
            '{"technology_fit":{"score":88,"matched":["Python"],"missing":["FastAPI"],"rationale":"Buen stack base"},"experience_fit":{"score":72,"matched":["Backend"],"missing":["Leadership"],"rationale":"Experiencia parcial"},"requirements_fit":{"score":70,"matched":["REST APIs"],"missing":["Microservices"],"rationale":"Cumple parcialmente"},"context_fit":{"score":95,"matched":["Salary","Location"],"missing":[],"rationale":"Buen contexto"},"red_flags":[{"type":"missing_technology","severity":"medium","detail":"Falta FastAPI"}],"summary":"Buen fit parcial"}'
        )
    ])

    def _fake_client(**kwargs):
        return _FakeAzureClient(response=response)

    monkeypatch.setattr(llm_module, "AzureOpenAI", _fake_client)

    service = AzureLLMService()
    result = service.evaluate_match_dimensions(
        {"professional_title": "Backend"},
        {"title": "Senior Backend"},
    )

    assert result["technology_fit"]["score"] == 88
    assert result["red_flags"][0]["severity"] == "medium"


def test_llm_service_empty_choices_returns_none(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

    def _fake_client(**kwargs):
        return _FakeAzureClient(response=_FakeCompletion([]))

    monkeypatch.setattr(llm_module, "AzureOpenAI", _fake_client)

    service = AzureLLMService()
    assert service.generate_match_feedback("cand", "vac", 92.5) is None
    assert service.evaluate_match_dimensions({}, {}) is None
    assert service.extract_candidate_profile("cv text") is None


def test_llm_service_generic_error_returns_none(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

    def _fake_client(**kwargs):
        return _FakeAzureClient(error=ValueError("any"))

    monkeypatch.setattr(llm_module, "AzureOpenAI", _fake_client)

    service = AzureLLMService()
    assert service.generate_match_feedback("cand", "vac", 92.5) is None
    assert service.evaluate_match_dimensions({}, {}) is None
    assert service.extract_candidate_profile("cv text") is None
