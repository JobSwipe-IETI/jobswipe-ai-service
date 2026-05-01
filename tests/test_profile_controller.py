from io import BytesIO

import app.controllers.profile_controller as profile_controller


class _StubPdfService:
    def __init__(self, text):
        self.text = text

    def extract_text_from_pdf(self, pdf_bytes):
        return self.text


class _StubCandidateProfile:
    def __init__(self, data):
        self._data = data

    def model_dump(self, by_alias=False, exclude_none=False):
        return self._data


class _StubLLMService:
    def __init__(self, profile=None):
        self.profile = profile

    def extract_candidate_profile(self, cv_text):
        return self.profile

    def analyze_candidate_profile(self, profile_dict):
        if profile_dict.get("professionalTitle") == "fail":
            return None
        return {
            "overall_score": 84,
            "summary": "Buen perfil",
            "sections": [],
            "suggestions": [],
        }


def test_extract_cv_endpoint_success(client, monkeypatch):
    monkeypatch.setattr(profile_controller, "pdf_service", _StubPdfService("cv text"))
    monkeypatch.setattr(
        profile_controller,
        "llm_service",
        _StubLLMService(
            _StubCandidateProfile(
                {
                    "professionalTitle": "Backend Engineer",
                    "summary": "Summary",
                    "skills": ["Python", "FastAPI"],
                    "experience": [
                        {
                            "role": "Backend Developer",
                            "project": "Order Management System",
                            "tech": ["Java", "Spring Boot"],
                        }
                    ],
                    "education": [
                        {
                            "institution": "ECI",
                            "degree": "Ingenieria de Sistemas",
                            "start": "2022",
                            "end": "2026",
                        }
                    ],
                }
            )
        ),
    )

    response = client.post(
        "/profiles/extract-cv",
        files={"file": ("cv.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["candidateProfile"]["professionalTitle"] == "Backend Engineer"
    assert response.json()["candidateProfile"]["experience"][0]["role"] == "Backend Developer"
    assert response.json()["usedAi"] is True


def test_extract_cv_endpoint_rejects_non_pdf(client):
    response = client.post(
        "/profiles/extract-cv",
        files={"file": ("cv.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 422
    assert "PDF" in response.json()["detail"]


def test_extract_cv_endpoint_rejects_empty_extraction(client, monkeypatch):
    monkeypatch.setattr(profile_controller, "pdf_service", _StubPdfService(""))

    response = client.post(
        "/profiles/extract-cv",
        files={"file": ("cv.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 422
    assert "No se pudo extraer texto" in response.json()["detail"]


def test_extract_cv_endpoint_returns_503_when_ai_fails(client, monkeypatch):
    monkeypatch.setattr(profile_controller, "pdf_service", _StubPdfService("cv text"))
    monkeypatch.setattr(profile_controller, "llm_service", _StubLLMService(None))

    response = client.post(
        "/profiles/extract-cv",
        files={"file": ("cv.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 503
    assert "Azure OpenAI" in response.json()["detail"]


def test_analyze_profile_endpoint_success(client, monkeypatch):
    monkeypatch.setattr(profile_controller, "llm_service", _StubLLMService())

    response = client.post(
        "/profiles/analyze",
        json={
            "professionalTitle": "Backend Engineer",
            "summary": "Summary",
            "skills": ["Python"],
            "experience": [],
            "education": [],
        },
    )

    assert response.status_code == 200
    assert response.json()["overall_score"] == 84


def test_analyze_profile_endpoint_returns_503_when_ai_fails(client, monkeypatch):
    monkeypatch.setattr(profile_controller, "llm_service", _StubLLMService())

    response = client.post(
        "/profiles/analyze",
        json={
            "professionalTitle": "fail",
            "summary": "Summary",
            "skills": ["Python"],
            "experience": [],
            "education": [],
        },
    )

    assert response.status_code == 503
    assert "No se pudo analizar" in response.json()["detail"]