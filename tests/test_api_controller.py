import app.controllers.embedding_controller as controller
import app.controllers.profile_controller as profile_controller


class _StubEmbeddingService:
    def __init__(self, vectors):
        self.vectors = vectors

    def generate_embedding(self, text):
        return self.vectors[text]


class _StubMatchingService:
    def __init__(self):
        self._similarity = 0.8

    def build_candidate_text(self, profile):
        return "candidate-structured-text"

    def build_vacancy_text(self, profile):
        return "vacancy-structured-text"

    def calculate_similarity(self, emb1, emb2):
        return self._similarity

    def to_compatibility_percentage(self, similarity):
        return 90.0

    def compatibility_level(self, percentage):
        return "high"

    def generate_rule_based_feedback(self, percentage):
        return "feedback from rules"


class _StubLLMService:
    def __init__(self, feedback=None):
        self.feedback = feedback

    def generate_match_feedback(self, candidate_text, vacancy_text, compatibility_percentage):
        return self.feedback


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "JobSwipe AI Service running"}


def test_generate_embedding_endpoint(client, monkeypatch):
    monkeypatch.setattr(
        controller,
        "service",
        _StubEmbeddingService({"hello": [0.1, 0.2]}),
    )

    response = client.post("/embeddings/", json={"text": "hello"})

    assert response.status_code == 200
    assert response.json() == {"result": [0.1, 0.2]}


def test_match_endpoint_with_text_input_and_rule_feedback(client, monkeypatch):
    monkeypatch.setattr(
        controller,
        "service",
        _StubEmbeddingService(
            {
                "cand": [1.0, 0.0],
                "vac": [1.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(controller, "matching_service", _StubMatchingService())
    monkeypatch.setattr(controller, "llm_service", _StubLLMService(feedback=None))

    response = client.post(
        "/embeddings/match",
        json={"candidate_text": "cand", "vacancy_text": "vac"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["compatibility_percentage"] == 90.0
    assert body["compatibility_level"] == "high"
    assert body["feedback"] == "feedback from rules"
    assert body["used_llm_feedback"] is False


def test_match_endpoint_with_structured_profiles(client, monkeypatch):
    monkeypatch.setattr(
        controller,
        "service",
        _StubEmbeddingService(
            {
                "candidate-structured-text": [1.0, 0.0],
                "vacancy-structured-text": [1.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(controller, "matching_service", _StubMatchingService())
    monkeypatch.setattr(controller, "llm_service", _StubLLMService(feedback="llm feedback"))

    response = client.post(
        "/embeddings/match",
        json={
            "candidateProfile": {"professionalTitle": "Backend"},
            "vacancyProfile": {"title": "Senior Backend"},
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["feedback"] == "llm feedback"
    assert body["used_llm_feedback"] is True


def test_match_endpoint_validation_error(client):
    response = client.post("/embeddings/match", json={})

    assert response.status_code == 422
    assert "candidate_text" in response.json()["detail"]


def test_extract_cv_endpoint_success(client, monkeypatch):
    class _StubPdfService:
        def extract_text_from_pdf(self, pdf_bytes):
            return "cv text"

    class _StubCandidateProfile:
        def model_dump(self, by_alias=False, exclude_none=False):
            return {
                "professionalTitle": "Backend Engineer",
                "summary": "Summary",
            }

    class _StubLLMService:
        def extract_candidate_profile(self, cv_text):
            return _StubCandidateProfile()

    monkeypatch.setattr(profile_controller, "pdf_service", _StubPdfService())
    monkeypatch.setattr(profile_controller, "llm_service", _StubLLMService())

    response = client.post(
        "/profiles/extract-cv",
        files={"file": ("cv.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["candidateProfile"]["professionalTitle"] == "Backend Engineer"
    assert body["usedAi"] is True
