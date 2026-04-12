from app.models.schemas import MatchRequest


def test_match_request_accepts_camel_case_payload():
    payload = {
        "candidateProfile": {
            "professionalTitle": "Backend Engineer",
            "expectedSalary": 7000000,
            "nationality": "Colombia",
            "sector": "Tecnologia",
            "skills": ["Python"],
        },
        "vacancyProfile": {
            "sector": "Tecnologia",
            "employmentType": "FULL_TIME",
            "experienceLevel": "SENIOR",
            "softSkills": ["Communication"],
            "technicalRequirements": ["REST APIs"],
            "minSalary": 5000000,
            "maxSalary": 9000000,
        },
    }

    req = MatchRequest.model_validate(payload)

    assert req.candidate_profile is not None
    assert req.candidate_profile.professional_title == "Backend Engineer"
    assert req.candidate_profile.expected_salary == 7000000
    assert req.candidate_profile.nationality == "Colombia"
    assert req.candidate_profile.sector == "Tecnologia"
    assert req.vacancy_profile is not None
    assert req.vacancy_profile.sector == "Tecnologia"
    assert req.vacancy_profile.employment_type == "FULL_TIME"
    assert req.vacancy_profile.experience_level == "SENIOR"
    assert req.vacancy_profile.soft_skills == ["Communication"]
    assert req.vacancy_profile.technical_requirements == ["REST APIs"]


def test_match_request_accepts_snake_case_payload():
    payload = {
        "candidate_profile": {"professional_title": "Data Engineer"},
        "vacancy_profile": {"experience_level": "MID"},
    }

    req = MatchRequest.model_validate(payload)

    assert req.candidate_profile.professional_title == "Data Engineer"
    assert req.vacancy_profile.experience_level == "MID"
