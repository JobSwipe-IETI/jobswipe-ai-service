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


def test_helper_branches_cover_zero_norm_and_fallback_paths():
    service = MatchingService()

    assert service._to_text(123) == "123"
    assert service._to_csv_text("  x  ") == "x"
    assert service._experience_to_text("  direct  ") == "direct"
    assert service._experience_to_text({"role": "Dev", "company": "Acme"}) == "Dev |  | Acme"
    assert service._education_to_text("  degree  ") == "degree"
    assert service._education_to_text({"institution": "ECI", "status": "En curso"}) == "ECI |  |  |  | En curso"
    assert service.calculate_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert service._normalize_similarity_score(0.35) == 0.0
    assert service._normalize_token("C# / .NET!!") == "c# .net"
    assert service._normalize_phrase("API REST") == "rest apis"
    assert service._value_to_list("a, b; c\n d") == ["a", "b", "c", "d"]
    assert service._candidate_soft_skills_blob({"summary": "S", "professional_title": "T"}).startswith("s")
    assert "github" in service._extract_candidate_technologies({"github": ["https://github.com/u"]})
    assert service._extract_vacancy_technologies({}) == set()
    assert service._extract_requirement_keywords({"description": "English required", "responsibilities": ["Code reviews"], "soft_skills": ["Communication"]})["hard"]
    assert service._candidate_matches_keyword("python fastapi", "api development") is False
    assert service._candidate_matches_soft_skill("trabajo en equipo", "teamwork") is True
    assert service._parse_year("2024-01-01") == 2024
    assert service._parse_year("unknown") is None
    assert service._experience_duration_years([{"start": "2020-01-01", "end": "2022-01-01"}]) == 2.0
    assert service._technology_match_score({}, {})[0] == 60.0


def test_scoring_and_penalty_branches_cover_llm_and_hard_requirements():
    service = MatchingService()

    candidate_profile = {
        "professional_title": "Senior Backend Engineer",
        "summary": "Backend engineer with 8 years in Python and FastAPI",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "experience": [
            {"role": "Senior Backend Engineer", "tech": ["Python", "FastAPI"], "description": "REST APIs", "start": "2018-01-01", "end": "Present"}
        ],
        "education": [{"institution": "ECI", "degree": "Systems", "start": "2010-01-01", "end": "2015-01-01", "status": "Graduated"}],
        "location": "Bogota",
        "languages": ["english", "spanish"],
        "sector": "Tecnologia",
        "expected_salary": 9000000,
    }
    vacancy_profile = {
        "title": "Senior Python Backend Engineer",
        "description": "English required. Build APIs and microservices. Code reviews. System design.",
        "location": "Bogota",
        "modality": "HYBRID",
        "experience_level": "SENIOR",
        "technologies": ["Python", "FastAPI", "PostgreSQL"],
        "technical_requirements": ["REST APIs", "Microservices"],
        "soft_skills": ["Communication", "Teamwork"],
        "responsibilities": ["Code reviews"],
        "min_salary": 8000000,
        "max_salary": 10000000,
    }

    assert service._infer_experience_level(candidate_profile) >= 3
    assert service._required_experience_level(vacancy_profile) == 3
    assert service._extract_languages(["Espanol", "English"]) == {"espanol", "english"}
    assert service._extract_required_languages(vacancy_profile) == {"english"}
    assert service._candidate_matches_keyword("python fastapi rest apis", "rest apis") is True
    assert service._candidate_matches_soft_skill("ownership collaboration", "teamwork") is True

    validation = service.validate_hard_requirements(candidate_profile, vacancy_profile)
    assert validation["passed"] is True
    assert validation["penalties"] == []

    structured = service.score_structured_match(
        candidate_profile,
        vacancy_profile,
        similarity_score=0.92,
        llm_evaluation={
            "technology_fit": {"score": 80, "matched": ["Python"], "missing": ["FastAPI"], "rationale": "ok"},
            "experience_fit": {"score": 70, "matched": ["Backend"], "missing": [], "rationale": "ok"},
            "requirements_fit": {"score": 65, "matched": ["REST APIs"], "missing": ["Microservices"], "rationale": "ok"},
            "context_fit": {"score": 90, "matched": ["Location"], "missing": [], "rationale": "ok"},
            "red_flags": [
                {"type": "missing_technology", "severity": "high", "detail": "Falta FastAPI"},
                {"type": "salary_gap", "severity": "medium", "detail": "No debe duplicar"},
                {"type": "experience_gap", "severity": "low", "detail": "No debe duplicar"},
            ],
            "summary": "Buen fit",
        },
    )

    assert structured["llm_evaluation_used"] is True
    assert structured["llm_red_flag_penalty"] > 0.0
    assert structured["details"]["requirements"]["source"] == "llm"
    assert structured["reasons"]

    fallback = service.score_structured_match(
        {"skills": ["Excel"], "experience": [], "location": "Medellin", "expected_salary": 20000000},
        {"technologies": ["Python"], "experience_level": "SENIOR", "location": "Bogota", "modality": "HYBRID", "max_salary": 8000000},
        similarity_score=0.2,
    )
    assert fallback["hard_requirements"]["passed"] is False
    assert fallback["details"]["requirements"]["source"] == "fallback"
    assert fallback["soft_matches"] == []


def test_feedback_levels_and_red_flags_cover_all_paths():
    service = MatchingService()

    assert service.to_compatibility_percentage(0.85) == 100.0
    assert service.compatibility_level(74.9) == "medium"
    assert service.generate_rule_based_feedback(80, ["a", "", None]).startswith("El perfil del candidato")

    assert service._normalize_dimension_score("x", 33.3) == 33.3
    assert service._normalize_llm_dimension({}, "technology_fit", {"score": 11.0, "matched": ["a"], "missing": ["b"], "rationale": "r"}) == {"score": 11.0, "matched": ["a"], "missing": ["b"], "rationale": "r"}
    assert service._normalize_red_flag_type("Missing Technology") == "missing_technologies"
    penalty, reasons = service._red_flag_penalty([{"type": "salary_gap", "severity": "high", "detail": "dup"}], [{"type": "salary_gap"}])
    assert penalty == 0.0


def test_edge_cases_with_malformed_data_cover_defensive_branches():
    """Test defensive branches handling non-dict items and malformed data."""
    service = MatchingService()

    # Lines 76-77: _experience_to_text with list containing non-dict items
    malformed_experience_list = ["string_item", 123, {"role": "Dev", "company": "Acme"}]
    result = service._experience_to_text(malformed_experience_list)
    assert "string_item" in result and "123" in result and "Dev" in result

    # Lines 103, 114-115: _education_to_text with list containing non-dict items
    malformed_education_list = ["string_item", 456, {"institution": "ECI", "degree": "CS"}]
    result = service._education_to_text(malformed_education_list)
    assert "string_item" in result and "456" in result and "ECI" in result

    # Test single string/number forms too (these test lines 70-72, 99-101)
    assert service._experience_to_text("single_string").strip() == "single_string"
    assert service._education_to_text("single_education").strip() == "single_education"

    # Lines 225, 245, 279: extract with empty/malformed structures
    candidate_empty = {
        "experience": [],
        "education": [],
        "skills": None,
        "github": "",
        "linkedin": "",
        "portfolio": None,
    }
    result = service._extract_candidate_technologies(candidate_empty)
    assert isinstance(result, set)
    assert len(result) == 0

    # Lines 332: extract_requirement_keywords with minimal fields
    vacancy_minimal = {"description": "No requirements", "responsibilities": None, "soft_skills": None}
    result = service._extract_requirement_keywords(vacancy_minimal)
    assert isinstance(result, dict)
    assert "hard" in result and "soft" in result

    # Lines 443, 489: language extraction with non-standard formats
    languages_messy = ["  English  ", "", None, "Spanish", "chinese", "PORTUGUESE"]
    result = service._extract_languages(languages_messy)
    assert isinstance(result, set)
    assert len(result) >= 3

    # Lines 515-520: _technology_match_score with empty vacancy tech
    candidate_profile = {"skills": ["Python", "FastAPI"], "experience": [], "education": []}
    vacancy_profile_empty = {"technologies": []}
    score_empty, breakdown = service._technology_match_score(candidate_profile, vacancy_profile_empty)
    assert score_empty == 60.0
    assert isinstance(breakdown, dict)
    
    # Lines 515-520: _technology_match_score with some matched
    vacancy_profile = {"technologies": ["Python", "Java"]}
    score, breakdown = service._technology_match_score(candidate_profile, vacancy_profile)
    assert isinstance(score, float)
    assert "matched" in breakdown and "missing" in breakdown

    # Lines 530, 577, 580, 608: score_structured_match with minimal data
    simple_candidate = {"skills": ["Python"], "professional_title": "Dev", "location": "Bogota"}
    simple_vacancy = {"technologies": ["Python"], "title": "Python Dev", "location": "Bogota"}
    score = service.score_structured_match(simple_candidate, simple_vacancy, similarity_score=0.5)
    assert isinstance(score, dict)
    assert "hard_requirements" in score
    assert "compatibility_percentage" in score
