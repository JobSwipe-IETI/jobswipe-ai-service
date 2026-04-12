from pydantic import BaseModel, ConfigDict, Field


class TextRequest(BaseModel):
    text: str


class CandidateExperienceInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str | None = Field(default=None, alias="role")
    project: str | None = Field(default=None, alias="project")
    company: str | None = Field(default=None, alias="company")
    tech: list[str] | None = Field(default=None, alias="tech")
    description: str | None = Field(default=None, alias="description")
    start: str | None = Field(default=None, alias="start")
    end: str | None = Field(default=None, alias="end")


class CandidateEducationInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    institution: str | None = Field(default=None, alias="institution")
    degree: str | None = Field(default=None, alias="degree")
    start: str | None = Field(default=None, alias="start")
    end: str | None = Field(default=None, alias="end")
    status: str | None = Field(default=None, alias="status")


class CandidateProfileInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    display_name: str | None = Field(default=None, alias="displayName")
    professional_title: str | None = Field(default=None, alias="professionalTitle")
    summary: str | None = Field(default=None, alias="summary")
    skills: list[str] | str | None = Field(default=None, alias="skills")
    experience: list[CandidateExperienceInput] | str | None = Field(default=None, alias="experience")
    education: list[CandidateEducationInput] | str | None = Field(default=None, alias="education")
    location: str | None = Field(default=None, alias="location")
    nationality: str | None = Field(default=None, alias="nationality")
    languages: list[str] | str | None = Field(default=None, alias="languages")
    sector: str | None = Field(default=None, alias="sector")
    expected_salary: float | None = Field(default=None, alias="expectedSalary")
    availability: str | None = Field(default=None, alias="availability")
    email: str | None = Field(default=None, alias="email")
    phone_number: str | None = Field(default=None, alias="phoneNumber")
    github: list[str] | str | None = Field(default=None, alias="github")
    linkedin: list[str] | str | None = Field(default=None, alias="linkedin")
    links: list[str] | None = Field(default=None, alias="links")


class VacancyProfileInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    title: str | None = None
    description: str | None = None
    location: str | None = None
    sector: str | None = None
    modality: str | None = None
    employment_type: str | None = Field(default=None, alias="employmentType")
    experience_level: str | None = Field(default=None, alias="experienceLevel")
    technologies: list[str] | None = None
    soft_skills: list[str] | None = Field(default=None, alias="softSkills")
    responsibilities: list[str] | None = None
    technical_requirements: list[str] | None = Field(
        default=None,
        alias="technicalRequirements",
    )
    min_salary: float | None = Field(default=None, alias="minSalary")
    max_salary: float | None = Field(default=None, alias="maxSalary")


class MatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    candidate_text: str | None = Field(default=None, alias="candidateText")
    vacancy_text: str | None = Field(default=None, alias="vacancyText")
    candidate_profile: CandidateProfileInput | None = Field(
        default=None,
        alias="candidateProfile",
    )
    vacancy_profile: VacancyProfileInput | None = Field(
        default=None,
        alias="vacancyProfile",
    )

    # Backward-compatible aliases for the original payload.
    text1: str | None = None
    text2: str | None = None


class MatchResponse(BaseModel):
    similarity_score: float
    compatibility_percentage: float
    compatibility_level: str
    feedback: str
    used_llm_feedback: bool
    score_breakdown: dict | None = None
