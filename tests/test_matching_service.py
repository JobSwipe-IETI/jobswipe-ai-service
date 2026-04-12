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


def test_context_score_rewards_matching_sector():
    service = MatchingService()

    candidate_profile = {
        "location": "Bogota",
        "sector": "Tecnologia",
        "languages": ["english"],
        "expected_salary": 8000000,
    }
    vacancy_profile = {
        "location": "Bogota",
        "sector": "Tecnologia",
        "modality": "HYBRID",
        "min_salary": 7000000,
        "max_salary": 9000000,
    }

    score, notes = service._context_score(candidate_profile, vacancy_profile)

    assert score > 60.0
    assert "sector alineado" in notes


def test_context_score_penalizes_different_sector():
    service = MatchingService()

    candidate_profile = {
        "sector": "Finanzas",
    }
    vacancy_profile = {
        "sector": "Tecnologia",
    }

    score, notes = service._context_score(candidate_profile, vacancy_profile)

    assert score == 54.0
    assert "sector diferente" in notes


def test_calculate_similarity_and_percentage_and_level():
    service = MatchingService()

    similarity = service.calculate_similarity([1.0, 0.0], [1.0, 0.0])
    assert similarity == 1.0

    assert service.to_compatibility_percentage(1.0) == 100.0
    assert service.to_compatibility_percentage(0.35) == 0.0
    assert service.compatibility_level(75.0) == "high"
    assert service.compatibility_level(50.0) == "medium"
    assert service.compatibility_level(49.99) == "low"


def test_rule_based_feedback_for_each_level():
    service = MatchingService()

    high = service.generate_rule_based_feedback(90)
    medium = service.generate_rule_based_feedback(70)
    low = service.generate_rule_based_feedback(40)

    assert "compatibilidad alta" in high
    assert "compatibilidad parcial" in medium
    assert "compatibilidad es baja" in low


def test_validate_hard_requirements_detects_missing_technologies_and_salary_gap():
    service = MatchingService()

    candidate_profile = {
        "skills": ["Excel"],
        "experience": [{"role": "Analyst", "start": "2024", "end": "2025"}],
        "location": "Medellin",
        "expected_salary": 12000000,
    }
    vacancy_profile = {
        "technologies": ["Python", "FastAPI", "PostgreSQL"],
        "experience_level": "SENIOR",
        "location": "Bogota",
        "modality": "HYBRID",
        "max_salary": 8000000,
    }

    validation = service.validate_hard_requirements(candidate_profile, vacancy_profile)

    assert validation["passed"] is False
    assert validation["penalty_points"] > 0
    assert any(item["type"] == "missing_technologies" for item in validation["penalties"])
    assert any(item["type"] == "salary_gap" for item in validation["penalties"])


def test_score_structured_match_rewards_good_profile_fit():
    service = MatchingService()

    candidate_profile = {
        "professional_title": "Senior Backend Engineer",
        "summary": "Backend engineer with 6 years building APIs in Python and FastAPI",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "experience": [
            {
                "role": "Senior Backend Engineer",
                "company": "Acme",
                "tech": ["Python", "FastAPI", "PostgreSQL"],
                "description": "6 years building REST APIs",
                "start": "2019",
                "end": "Present",
            }
        ],
        "location": "Bogota",
        "languages": ["english", "spanish"],
        "expected_salary": 8000000,
    }
    vacancy_profile = {
        "title": "Senior Python Backend Engineer",
        "description": "Need strong Python, FastAPI and PostgreSQL skills. English required.",
        "location": "Bogota",
        "modality": "HYBRID",
        "experience_level": "SENIOR",
        "technologies": ["Python", "FastAPI", "PostgreSQL"],
        "technical_requirements": ["English", "REST APIs"],
        "min_salary": 7000000,
        "max_salary": 9000000,
    }

    score = service.score_structured_match(candidate_profile, vacancy_profile, similarity_score=0.78)

    assert score["compatibility_percentage"] >= 70.0
    assert score["hard_requirements"]["passed"] is True
    assert score["technology_score"] == 100.0
    assert "build apis" not in score["details"]["requirements"]["missing"]
    assert score["details"]["requirements"]["hard"]["matched"]


def test_requirements_extraction_separates_hard_and_soft_signals():
    service = MatchingService()

    vacancy_profile = {
        "description": "Buscamos backend engineer con English required y experiencia en microservices.",
        "responsibilities": ["Build APIs", "Code reviews", "System design"],
        "technical_requirements": ["REST APIs", "FastAPI"],
        "soft_skills": ["Communication", "Ownership"],
        "technologies": ["Python", "PostgreSQL"],
    }

    requirements = service._extract_requirement_keywords(vacancy_profile)

    assert "python" in requirements["hard"]
    assert "api development" in requirements["hard"]
    assert "communication" in requirements["soft"]
    assert "ownership" in requirements["soft"]
    assert "buscamos backend engineer con english required y experiencia en microservices." not in requirements["hard"]


def test_semantic_score_is_capped_when_technical_fit_is_low():
    service = MatchingService()

    candidate_profile = {
        "professional_title": "Data Analyst",
        "summary": "Python para analitica y reportes",
        "skills": ["Python", "SQL"],
        "experience": [
            {
                "role": "Data Analyst",
                "tech": ["Python", "SQL"],
                "description": "Reportes y dashboards",
                "start": "2021",
                "end": "Present",
            }
        ],
        "location": "Bogota",
        "languages": ["english"],
        "expected_salary": 8000000,
    }
    vacancy_profile = {
        "title": "Senior Python Backend Engineer",
        "description": "English required. Build APIs and microservices.",
        "location": "Bogota",
        "modality": "HYBRID",
        "experience_level": "SENIOR",
        "technologies": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "technical_requirements": ["REST APIs", "Microservices"],
        "soft_skills": ["Communication"],
        "max_salary": 9000000,
    }

    score = service.score_structured_match(candidate_profile, vacancy_profile, similarity_score=0.95)

    assert score["technology_score"] == 25.0
    assert score["semantic_score"] == 55.0
    assert score["compatibility_percentage"] < 40.0


def test_score_structured_match_uses_llm_dimension_scores_when_available():
    service = MatchingService()

    candidate_profile = {
        "professional_title": "Backend Developer",
        "skills": ["Python", "PostgreSQL"],
        "experience": [{"role": "Backend Developer", "start": "2021", "end": "Present"}],
        "location": "Bogota",
        "languages": ["english"],
        "expected_salary": 8000000,
    }
    vacancy_profile = {
        "title": "Senior Python Backend Engineer",
        "location": "Bogota",
        "modality": "HYBRID",
        "experience_level": "SENIOR",
        "technologies": ["Python", "FastAPI", "PostgreSQL"],
        "technical_requirements": ["REST APIs", "Microservices"],
        "max_salary": 9000000,
    }
    llm_evaluation = {
        "technology_fit": {
            "score": 78,
            "matched": ["Python", "PostgreSQL"],
            "missing": ["FastAPI"],
            "rationale": "Buen stack base",
        },
        "experience_fit": {
            "score": 62,
            "matched": ["Backend development"],
            "missing": ["Senior ownership"],
            "rationale": "Experiencia parcial",
        },
        "requirements_fit": {
            "score": 58,
            "matched": ["REST APIs"],
            "missing": ["Microservices"],
            "rationale": "Cumple parcialmente",
        },
        "context_fit": {
            "score": 92,
            "matched": ["Location", "Salary"],
            "missing": [],
            "rationale": "Buen contexto",
        },
        "red_flags": [{"type": "missing_technology", "severity": "medium", "detail": "Falta FastAPI"}],
        "summary": "Buen fit parcial",
    }

    score = service.score_structured_match(
        candidate_profile,
        vacancy_profile,
        similarity_score=0.91,
        llm_evaluation=llm_evaluation,
    )

    assert score["llm_evaluation_used"] is True
    assert score["technology_score"] == 78.0
    assert score["experience_score"] == 62.0
    assert score["llm_red_flag_penalty"] == 0.0
    assert score["details"]["requirements"]["source"] == "llm"
    assert score["details"]["requirements"]["fallback_assessment"] is not None
    assert score["details"]["requirements"]["hard"]["matched"] == ["REST APIs"]


def test_llm_red_flag_does_not_double_penalize_hard_requirement():
    service = MatchingService()

    candidate_profile = {
        "professional_title": "Backend Developer",
        "skills": ["Python", "PostgreSQL"],
        "experience": [{"role": "Backend Developer", "start": "2021", "end": "Present"}],
        "location": "Bogota",
        "expected_salary": 8000000,
    }
    vacancy_profile = {
        "title": "Senior Python Backend Engineer",
        "location": "Bogota",
        "modality": "HYBRID",
        "experience_level": "MID",
        "technologies": ["Python", "FastAPI", "PostgreSQL"],
        "max_salary": 9000000,
    }
    llm_evaluation = {
        "technology_fit": {
            "score": 80,
            "matched": ["Python", "PostgreSQL"],
            "missing": ["FastAPI"],
            "rationale": "Buen stack base",
        },
        "experience_fit": {"score": 70, "matched": [], "missing": [], "rationale": ""},
        "requirements_fit": {"score": 70, "matched": [], "missing": [], "rationale": ""},
        "context_fit": {"score": 90, "matched": [], "missing": [], "rationale": ""},
        "red_flags": [{"type": "missing_technology", "severity": "high", "detail": "Falta FastAPI"}],
        "summary": "Buen fit parcial",
    }

    score = service.score_structured_match(
        candidate_profile,
        vacancy_profile,
        similarity_score=0.9,
        llm_evaluation=llm_evaluation,
    )

    assert any(
        item["type"] == "missing_technologies"
        for item in score["hard_requirements"]["penalties"]
    )
    assert score["llm_red_flag_penalty"] == 0.0


def test_requirements_details_use_fallback_when_llm_not_available():
    service = MatchingService()

    candidate_profile = {
        "professional_title": "Backend Engineer",
        "skills": ["Python", "FastAPI", "PostgreSQL"],
        "experience": [{"role": "Backend Engineer", "description": "REST APIs", "start": "2020", "end": "Present"}],
        "location": "Bogota",
        "languages": ["english"],
        "expected_salary": 8000000,
    }
    vacancy_profile = {
        "title": "Senior Python Backend Engineer",
        "description": "English required.",
        "location": "Bogota",
        "modality": "HYBRID",
        "experience_level": "MID",
        "technologies": ["Python", "FastAPI", "PostgreSQL"],
        "technical_requirements": ["REST APIs"],
        "soft_skills": ["Communication"],
        "max_salary": 9000000,
    }

    score = service.score_structured_match(
        candidate_profile,
        vacancy_profile,
        similarity_score=0.88,
    )

    assert score["details"]["requirements"]["source"] == "fallback"
    assert score["details"]["requirements"]["fallback_assessment"] is None
