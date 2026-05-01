import app.controllers.embedding_controller as embedding_controller


class _StubEmbeddingService:
    def generate_embedding(self, text):
        return [1.0, 0.0] if text else [0.0, 1.0]


class _StubMatchingService:
    def build_candidate_text(self, payload):
        return f"candidate:{payload.get('professional_title', '')}"

    def build_vacancy_text(self, payload):
        return f"vacancy:{payload.get('title', '')}"

    def calculate_similarity(self, candidate_embedding, vacancy_embedding):
        return 0.91

    def score_structured_match(self, **kwargs):
        return {
            "compatibility_percentage": 88.0,
            "reasons": ["Buen fit"],
        }

    def to_compatibility_percentage(self, similarity_score):
        return 77.0

    def compatibility_level(self, compatibility_percentage):
        return "high"

    def generate_rule_based_feedback(self, compatibility_percentage, reasons=None):
        return f"Fallback {compatibility_percentage}"


class _StubLLMService:
    def __init__(self, feedback="LLM feedback"):
        self.feedback = feedback

    def evaluate_match_dimensions(self, candidate_profile, vacancy_profile):
        return {"technology_fit": {"score": 80}}

    def generate_match_feedback(self, candidate_text, vacancy_text, compatibility_percentage):
        return self.feedback


class _StubProfile:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class _StubMatchRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_generate_embedding_endpoint_uses_service(monkeypatch):
    monkeypatch.setattr(embedding_controller, "service", _StubEmbeddingService())

    result = embedding_controller.generate_embedding(type("Req", (), {"text": "hola"})())

    assert result == {"result": [1.0, 0.0]}


def test_match_texts_uses_structured_profiles(monkeypatch):
    monkeypatch.setattr(embedding_controller, "service", _StubEmbeddingService())
    monkeypatch.setattr(embedding_controller, "matching_service", _StubMatchingService())
    monkeypatch.setattr(embedding_controller, "llm_service", _StubLLMService())

    request = _StubMatchRequest(
        candidate_text=None,
        vacancy_text=None,
        text1=None,
        text2=None,
        candidate_profile=_StubProfile({"professional_title": "Backend Engineer"}),
        vacancy_profile=_StubProfile({"title": "Senior Backend"}),
    )

    response = embedding_controller.match_texts(request)

    assert response.compatibility_percentage == 88.0
    assert response.used_llm_feedback is True
    assert response.score_breakdown["compatibility_percentage"] == 88.0


def test_match_texts_falls_back_to_text_inputs_and_422(monkeypatch):
    monkeypatch.setattr(embedding_controller, "service", _StubEmbeddingService())
    monkeypatch.setattr(embedding_controller, "matching_service", _StubMatchingService())
    monkeypatch.setattr(embedding_controller, "llm_service", _StubLLMService(feedback=""))

    request = _StubMatchRequest(
        candidate_text="candidate text",
        vacancy_text="vacancy text",
        text1=None,
        text2=None,
        candidate_profile=None,
        vacancy_profile=None,
    )

    response = embedding_controller.match_texts(request)

    assert response.compatibility_percentage == 77.0
    assert response.used_llm_feedback is False
    assert response.feedback == "Fallback 77.0"

    invalid_request = _StubMatchRequest(
        candidate_text=None,
        vacancy_text=None,
        text1=None,
        text2=None,
        candidate_profile=None,
        vacancy_profile=None,
    )

    try:
        embedding_controller.match_texts(invalid_request)
        assert False, "Expected HTTPException"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422


def test_match_texts_loads_profiles_when_text_and_profile_both_sent(monkeypatch):
    """Test lines 42-45: profile payloads loaded when both text and profile are sent."""
    monkeypatch.setattr(embedding_controller, "service", _StubEmbeddingService())
    monkeypatch.setattr(embedding_controller, "matching_service", _StubMatchingService())
    monkeypatch.setattr(embedding_controller, "llm_service", _StubLLMService(feedback=""))

    # Send both texts AND profiles - forces lines 42-45 execution
    request = _StubMatchRequest(
        candidate_text="candidate text",
        vacancy_text="vacancy text",
        text1=None,
        text2=None,
        candidate_profile=_StubProfile({"professional_title": "Backend Engineer"}),
        vacancy_profile=_StubProfile({"title": "Senior Backend"}),
    )

    response = embedding_controller.match_texts(request)
    
    # Verify the response works with both paths (profiles are loaded and used for breakdown)
    assert response.compatibility_percentage == 88.0
    assert response.used_llm_feedback is False
