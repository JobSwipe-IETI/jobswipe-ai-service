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


class _FakeResponsesApi:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


class _FakeProjectClient:
    def __init__(self, response=None, error=None):
        self.responses = _FakeResponsesApi(response=response, error=error)


def _configured_service(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    return AzureLLMService()


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


def test_llm_service_internal_helpers_cover_static_feedback(monkeypatch):
    service = AzureLLMService()

    assert service._normalize_text("  hola  ") == "hola"
    assert service._normalize_text(None) == ""
    assert service._normalize_list(None) == []
    assert service._normalize_list([" a ", "", 5]) == [" a ", "", 5]
    assert service._normalize_list("a, b , ,c") == ["a", "b", "c"]
    assert service._has_value(["", None]) is False
    assert service._has_value(["x", None]) is True

    suggestion = service._create_profile_suggestion(service.PROFILE_SECTION_SKILLS, "Msg", "high", "Ejemplo")
    assert suggestion["category"] == service.PROFILE_SECTION_SKILLS
    assert suggestion["example"] == "Ejemplo"

    assert service._profile_section_score([]) == 95
    assert service._profile_section_score([
        {"severity": "high"},
        {"severity": "medium"},
        {"severity": "medium"},
    ]) == 59

    sections = service._build_profile_sections([
        {"category": service.PROFILE_SECTION_GENERAL, "severity": "low", "message": "A"},
        {"category": "Otros", "severity": "medium", "message": "B"},
    ])
    assert sections[0]["name"] == service.PROFILE_SECTION_GENERAL
    assert any(item["name"] == "Otros" for item in sections)

    general = service._build_general_profile_suggestions({"professionalTitle": "", "summary": "corto"})
    assert len(general) == 2

    skills = service._build_skills_profile_suggestions({
        "skills": ["Trabajo en equipo"],
        "experience": [{"tech": ["Python", "FastAPI", "Python"]}, "ignored"],
    })
    assert len(skills) == 2

    experience = service._build_experience_profile_suggestions({"experience": [{"role": "Dev"}]})
    assert len(experience) == 1
    assert service._build_experience_profile_suggestions({"experience": []})[0]["severity"] == "high"

    education = service._build_education_profile_suggestions({"education": [{"institution": "ECI"}]})
    assert len(education) == 1

    additional = service._build_additional_profile_suggestions({})
    assert len(additional) == 2

    static_suggestions, overall_score, short_summary = service._build_static_profile_feedback({})
    assert static_suggestions
    assert overall_score < 65
    assert "Perfil con varias áreas" in short_summary

    system_prompt, user_prompt = service._build_profile_prompts({"professionalTitle": "Backend"})
    assert "Devuelve SOLO JSON valido" in system_prompt
    assert "candidateProfile" in user_prompt

    loaded = service._safe_json_loads("```json\n{\"a\": 1}\n```")
    assert loaded == {"a": 1}
    assert service._safe_json_loads("not json") == {}


def test_llm_service_merges_ai_feedback_and_normalizes_sector(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

    response = _FakeCompletion([
        _FakeChoice(
            '{"overall_score": 88, "summary": "AI summary", "categories": ["x"], '
            '"sections": [{"name": "Habilidades", "suggestions": [{"message": "AI skill", "severity": "medium"}]}], '
            '"metadata": {"model": "fake"}, "suggestions": [{"message": "Flat", "severity": "low"}]}'
        )
    ])

    def _fake_client(**kwargs):
        return _FakeAzureClient(response=response)

    monkeypatch.setattr(llm_module, "AzureOpenAI", _fake_client)

    service = AzureLLMService()
    profile = {
        "professionalTitle": "Backend Engineer",
        "summary": "This summary is long enough to avoid the static high severity branch.",
        "skills": ["Python"],
        "experience": [{"role": "Backend Developer", "tech": ["Python"]}],
        "education": [{"institution": "ECI", "degree": "Systems"}],
        "location": "Bogota",
        "nationality": "Colombia",
        "languages": ["Espanol"],
        "sector": "tech",
        "expectedSalary": 7000000,
        "availability": "Immediate",
        "email": "john@example.com",
        "phoneNumber": "123",
        "github": ["https://github.com/user"],
        "linkedin": ["https://linkedin.com/in/user"],
    }

    result = service.analyze_candidate_profile(profile)

    assert result["overall_score"] == 88
    assert result["summary"] == "AI summary"
    assert result["categories"] == ["x"]
    assert result["metadata"] == {"model": "fake"}
    assert any(section["name"] == "Habilidades" for section in result["sections"])
    assert service._normalize_extracted_profile({"sector": "tech"})["sector"] == "Tecnologia"
    assert service._normalize_sector("unknown") == "Otro"


def test_llm_service_project_responses_api_paths(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com/api/projects/demo")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

    project_client = _FakeProjectClient(
        response=type("Resp", (), {"output_text": '{"overall_score": 91, "summary": "Project AI", "sections": [], "metadata": {"model": "project"}}'})()
    )

    def _fake_openai(**kwargs):
        return project_client

    monkeypatch.setattr(llm_module, "OpenAI", _fake_openai)

    service = AzureLLMService()
    assert service.is_configured() is True
    assert service._use_project_responses_api is True

    feedback = service.generate_match_feedback("cand", "vac", 87.0)
    assert feedback == '{"overall_score": 91, "summary": "Project AI", "sections": [], "metadata": {"model": "project"}}'

    dimensions = service.evaluate_match_dimensions({"skills": ["Python"]}, {"technologies": ["Python"]})
    assert dimensions == {"overall_score": 91, "summary": "Project AI", "sections": [], "metadata": {"model": "project"}}

    profile = service.extract_candidate_profile("cv text")
    assert isinstance(profile, CandidateProfileInput)
    assert profile.summary == "Project AI"

