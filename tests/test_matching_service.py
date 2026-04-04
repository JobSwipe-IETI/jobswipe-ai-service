from app.services.matching_service import MatchingService


def test_to_csv_text_handles_none_list_and_string():
    service = MatchingService()

    assert service._to_csv_text(None) == ""
    assert service._to_csv_text([" Python ", "", "FastAPI"]) == "Python, FastAPI"
    assert service._to_csv_text("  backend  ") == "backend"


def test_build_candidate_text_contains_expected_sections():
    service = MatchingService()
    payload = {
        "professional_title": "Backend Engineer",
        "summary": "Resumen",
        "skills": ["Python", "FastAPI"],
        "experience": [
            {
                "role": "Backend Developer",
                "project": "Order Management System",
                "tech": ["Java", "Spring Boot"],
                "description": "APIs REST",
            }
        ],
        "education": [
            {
                "institution": "ECI",
                "degree": "Systems",
                "start": "2022",
                "end": "2026",
            }
        ],
        "location": "Bogota",
        "languages": ["es", "en"],
        "expected_salary": 7000000,
        "availability": "Immediate",
        "email": "persona@email.com",
        "phone_number": "+57 300 000 0000",
    }

    text = service.build_candidate_text(payload)

    assert "Titulo profesional: Backend Engineer" in text
    assert "Skills: Python, FastAPI" in text
    assert "Salario esperado: 7000000" in text
    assert "Backend Developer | Order Management System | Java, Spring Boot | APIs REST" in text
    assert "ECI | Systems | 2022 | 2026" in text


def test_build_vacancy_text_contains_expected_sections():
    service = MatchingService()
    payload = {
        "title": "Senior Backend",
        "description": "Role description",
        "location": "Bogota",
        "modality": "HYBRID",
        "employment_type": "FULL_TIME",
        "experience_level": "SENIOR",
        "technologies": ["Python", "FastAPI"],
        "soft_skills": ["Ownership"],
        "responsibilities": ["Build APIs"],
        "technical_requirements": ["REST"],
        "min_salary": 6000000,
        "max_salary": 9000000,
    }

    text = service.build_vacancy_text(payload)

    assert "Titulo: Senior Backend" in text
    assert "Tecnologias: Python, FastAPI" in text
    assert "Rango salarial: 6000000 - 9000000" in text


def test_calculate_similarity_and_percentage_and_level():
    service = MatchingService()

    similarity = service.calculate_similarity([1.0, 0.0], [1.0, 0.0])
    assert similarity == 1.0

    assert service.to_compatibility_percentage(1.0) == 100.0
    assert service.to_compatibility_percentage(-1.0) == 0.0
    assert service.compatibility_level(85.0) == "high"
    assert service.compatibility_level(60.0) == "medium"
    assert service.compatibility_level(59.99) == "low"


def test_rule_based_feedback_for_each_level():
    service = MatchingService()

    high = service.generate_rule_based_feedback(90)
    medium = service.generate_rule_based_feedback(70)
    low = service.generate_rule_based_feedback(40)

    assert "alineado" in high
    assert "compatibilidad parcial" in medium
    assert "compatibilidad es baja" in low
