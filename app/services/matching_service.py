import re
from datetime import datetime

import numpy as np


class MatchingService:
    _TECH_SYNONYMS = {
        "js": "javascript",
        "node": "node.js",
        "nodejs": "node.js",
        "postgres": "postgresql",
        "postgre": "postgresql",
        "py": "python",
        "golang": "go",
        "csharp": "c#",
        "dotnet": ".net",
    }

    _EXPERIENCE_LEVEL_ORDER = {
        "intern": 0,
        "trainee": 0,
        "junior": 1,
        "jr": 1,
        "semi senior": 2,
        "semisenior": 2,
        "mid": 2,
        "middle": 2,
        "senior": 3,
        "sr": 3,
        "lead": 4,
        "staff": 4,
        "principal": 5,
    }

    _SOFT_SKILL_ALIASES = {
        "communication": ["communication", "comunicacion", "communicator"],
        "ownership": ["ownership", "accountability", "responsibility", "responsable"],
        "leadership": ["leadership", "liderazgo", "mentoring"],
        "teamwork": ["teamwork", "collaboration", "collaborative", "trabajo en equipo"],
    }

    _REQUIREMENT_ALIASES = {
        "rest apis": ["rest api", "rest apis", "api rest", "apis rest", "backend api"],
        "microservices": ["microservices", "microservicios", "distributed systems"],
        "system design": ["system design", "arquitectura", "software architecture"],
        "code reviews": ["code review", "code reviews", "peer review"],
        "api development": ["build apis", "api development", "develop apis", "desarrollo de apis"],
        "english": ["english", "ingles", "bilingual"],
    }

    def _to_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _to_csv_text(self, value: list[str] | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(cleaned)
        return str(value).strip()

    def _experience_to_text(self, experience) -> str:
        if experience is None:
            return ""
        if isinstance(experience, str):
            return experience.strip()
        if isinstance(experience, list):
            entries = []
            for item in experience:
                if not isinstance(item, dict):
                    entries.append(self._to_text(item))
                    continue
                parts = [
                    item.get("role"),
                    item.get("project"),
                    item.get("company"),
                    self._to_csv_text(item.get("tech")),
                    item.get("description"),
                    item.get("start"),
                    item.get("end"),
                ]
                cleaned = [self._to_text(part) for part in parts if self._to_text(part)]
                if cleaned:
                    entries.append(" | ".join(cleaned))
            return "\n".join(entries)
        if isinstance(experience, dict):
            return " | ".join(
                [
                    self._to_text(experience.get("role")),
                    self._to_text(experience.get("project")),
                    self._to_text(experience.get("company")),
                    self._to_csv_text(experience.get("tech")),
                    self._to_text(experience.get("description")),
                    self._to_text(experience.get("start")),
                    self._to_text(experience.get("end")),
                ]
            ).strip(" |")
        return self._to_text(experience)

    def _education_to_text(self, education) -> str:
        if education is None:
            return ""
        if isinstance(education, str):
            return education.strip()
        if isinstance(education, list):
            entries = []
            for item in education:
                if not isinstance(item, dict):
                    entries.append(self._to_text(item))
                    continue
                parts = [
                    item.get("institution"),
                    item.get("degree"),
                    item.get("start"),
                    item.get("end"),
                    item.get("status"),
                ]
                cleaned = [self._to_text(part) for part in parts if self._to_text(part)]
                if cleaned:
                    entries.append(" | ".join(cleaned))
            return "\n".join(entries)
        if isinstance(education, dict):
            return " | ".join(
                [
                    self._to_text(education.get("institution")),
                    self._to_text(education.get("degree")),
                    self._to_text(education.get("start")),
                    self._to_text(education.get("end")),
                    self._to_text(education.get("status")),
                ]
            ).strip(" |")
        return self._to_text(education)

    def build_candidate_text(self, candidate_profile: dict) -> str:
        parts = [
            f"Titulo profesional: {candidate_profile.get('professional_title', '')}",
            f"Resumen: {candidate_profile.get('summary', '')}",
            f"Skills: {self._to_csv_text(candidate_profile.get('skills'))}",
            f"Experiencia: {self._experience_to_text(candidate_profile.get('experience'))}",
            f"Educacion: {self._education_to_text(candidate_profile.get('education'))}",
            f"Ubicacion: {candidate_profile.get('location', '')}",
            f"Idiomas: {self._to_csv_text(candidate_profile.get('languages'))}",
            f"Salario esperado: {candidate_profile.get('expected_salary', '')}",
            f"Disponibilidad: {candidate_profile.get('availability', '')}",
            f"Email: {candidate_profile.get('email', '')}",
            f"Telefono: {candidate_profile.get('phone_number', '')}",
            f"Github: {candidate_profile.get('github', '')}",
            f"LinkedIn: {candidate_profile.get('linkedin', '')}",
        ]
        return "\n".join(parts)

    def build_vacancy_text(self, vacancy_profile: dict) -> str:
        parts = [
            f"Titulo: {vacancy_profile.get('title', '')}",
            f"Descripcion: {vacancy_profile.get('description', '')}",
            f"Ubicacion: {vacancy_profile.get('location', '')}",
            f"Modalidad: {vacancy_profile.get('modality', '')}",
            f"Tipo de contrato: {vacancy_profile.get('employment_type', '')}",
            f"Nivel de experiencia: {vacancy_profile.get('experience_level', '')}",
            f"Tecnologias: {self._to_csv_text(vacancy_profile.get('technologies'))}",
            f"Habilidades blandas: {self._to_csv_text(vacancy_profile.get('soft_skills'))}",
            f"Responsabilidades: {self._to_csv_text(vacancy_profile.get('responsibilities'))}",
            (
                "Requisitos tecnicos: "
                f"{self._to_csv_text(vacancy_profile.get('technical_requirements'))}"
            ),
            (
                "Rango salarial: "
                f"{vacancy_profile.get('min_salary', '')} - {vacancy_profile.get('max_salary', '')}"
            ),
        ]
        return "\n".join(parts)

    def calculate_similarity(self, embedding1: list, embedding2: list) -> float:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)

        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        return float(similarity)

    def _clamp(self, value: float, lower: float = 0.0, upper: float = 100.0) -> float:
        return max(lower, min(upper, value))

    def _normalize_similarity_score(self, similarity_score: float) -> float:
        # Cosine similarities in embedding-based matching are usually positive.
        # This calibration avoids treating a merely related profile as a strong match.
        lower_bound = 0.35
        upper_bound = 0.85
        normalized = ((similarity_score - lower_bound) / (upper_bound - lower_bound)) * 100.0
        return round(self._clamp(normalized), 2)

    def _normalize_token(self, token: str) -> str:
        cleaned = re.sub(r"[^a-z0-9+#.\- ]+", " ", token.lower()).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return self._TECH_SYNONYMS.get(cleaned, cleaned)

    def _normalize_phrase(self, phrase: str) -> str:
        normalized = self._normalize_token(phrase)
        for canonical, aliases in self._REQUIREMENT_ALIASES.items():
            if normalized == canonical or normalized in aliases:
                return canonical
        return normalized

    def _value_to_list(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [self._to_text(item) for item in value if self._to_text(item)]
        if isinstance(value, str):
            parts = re.split(r"[,;/\n]+", value)
            return [part.strip() for part in parts if part.strip()]
        return [self._to_text(value)] if self._to_text(value) else []

    def _candidate_soft_skills_blob(self, candidate_profile: dict) -> str:
        return " ".join(
            [
                self._to_text(candidate_profile.get("summary")),
                self._experience_to_text(candidate_profile.get("experience")),
                self._to_text(candidate_profile.get("professional_title")),
            ]
        ).lower()

    def _extract_candidate_technologies(self, candidate_profile: dict) -> set[str]:
        tokens = set()
        for item in self._value_to_list(candidate_profile.get("skills")):
            tokens.add(self._normalize_token(item))

        experience = candidate_profile.get("experience")
        if isinstance(experience, list):
            for item in experience:
                if not isinstance(item, dict):
                    continue
                for tech in self._value_to_list(item.get("tech")):
                    tokens.add(self._normalize_token(tech))

        for link in self._value_to_list(candidate_profile.get("github")):
            if "github" in link.lower():
                tokens.add("github")

        return {token for token in tokens if token}

    def _extract_vacancy_technologies(self, vacancy_profile: dict) -> set[str]:
        tokens = set()
        for item in self._value_to_list(vacancy_profile.get("technologies")):
            tokens.add(self._normalize_token(item))
        return {token for token in tokens if token}

    def _extract_requirement_keywords(self, vacancy_profile: dict) -> dict:
        hard_keywords = set()
        soft_keywords = set()

        tech_keywords = self._extract_vacancy_technologies(vacancy_profile)
        hard_keywords.update(tech_keywords)

        for item in self._value_to_list(vacancy_profile.get("technical_requirements")):
            normalized = self._normalize_phrase(item)
            if normalized:
                hard_keywords.add(normalized)

        for item in self._value_to_list(vacancy_profile.get("responsibilities")):
            normalized = self._normalize_phrase(item)
            if normalized:
                if normalized in {"api development", "code reviews", "system design"}:
                    hard_keywords.add(normalized)
                else:
                    soft_keywords.add(normalized)

        for item in self._value_to_list(vacancy_profile.get("soft_skills")):
            normalized = self._normalize_phrase(item)
            if normalized:
                soft_keywords.add(normalized)

        description_blob = " ".join(self._value_to_list(vacancy_profile.get("description"))).lower()
        description_checks = [
            "english",
            "rest apis",
            "microservices",
            "system design",
            "code reviews",
            "api development",
        ]
        for keyword in description_checks:
            aliases = self._REQUIREMENT_ALIASES.get(keyword, [keyword])
            if any(alias in description_blob for alias in aliases):
                hard_keywords.add(keyword)

        return {
            "hard": {keyword for keyword in hard_keywords if keyword},
            "soft": {keyword for keyword in soft_keywords if keyword},
        }

    def _candidate_context_blob(self, candidate_profile: dict) -> str:
        return " ".join(
            [
                self._to_text(candidate_profile.get("professional_title")),
                self._to_text(candidate_profile.get("summary")),
                self._experience_to_text(candidate_profile.get("experience")),
                self._education_to_text(candidate_profile.get("education")),
                self._to_csv_text(candidate_profile.get("languages")),
                self._to_csv_text(candidate_profile.get("skills")),
            ]
        ).lower()

    def _infer_experience_level(self, candidate_profile: dict) -> int:
        text = self._candidate_context_blob(candidate_profile)

        highest_level = 1
        for label, level in self._EXPERIENCE_LEVEL_ORDER.items():
            if label in text:
                highest_level = max(highest_level, level)

        year_matches = re.findall(r"(\d+)\s*(?:\+)?\s*(?:anos|año|years?)", text)
        years = max((int(match) for match in year_matches), default=0)
        if years >= 8:
            highest_level = max(highest_level, 4)
        elif years >= 5:
            highest_level = max(highest_level, 3)
        elif years >= 2:
            highest_level = max(highest_level, 2)

        experience = candidate_profile.get("experience")
        if isinstance(experience, list):
            highest_level = max(highest_level, min(len(experience), 4))

        return highest_level

    def _required_experience_level(self, vacancy_profile: dict) -> int:
        level = self._normalize_token(self._to_text(vacancy_profile.get("experience_level")))
        return self._EXPERIENCE_LEVEL_ORDER.get(level, 1)

    def _extract_languages(self, value) -> set[str]:
        languages = set()
        for item in self._value_to_list(value):
            normalized = self._normalize_token(item)
            if normalized:
                languages.add(normalized)
        return languages

    def _extract_required_languages(self, vacancy_profile: dict) -> set[str]:
        languages = set()
        sources = [
            vacancy_profile.get("technical_requirements"),
            vacancy_profile.get("description"),
        ]
        for source in sources:
            for item in self._value_to_list(source):
                normalized = self._normalize_token(item)
                for language in ("english", "ingles", "spanish", "espanol"):
                    if language in normalized:
                        languages.add(language)
        return languages

    def _candidate_matches_keyword(self, candidate_blob: str, keyword: str) -> bool:
        if not keyword:
            return False
        aliases = self._REQUIREMENT_ALIASES.get(keyword, [keyword])
        return any(alias in candidate_blob for alias in aliases)

    def _candidate_matches_soft_skill(self, candidate_blob: str, keyword: str) -> bool:
        aliases = self._SOFT_SKILL_ALIASES.get(keyword, [keyword])
        return any(alias in candidate_blob for alias in aliases)

    def _parse_year(self, value) -> int | None:
        text = self._to_text(value)
        match = re.search(r"(19|20)\d{2}", text)
        if not match:
            return None
        return int(match.group(0))

    def _experience_duration_years(self, experience) -> float:
        if not isinstance(experience, list):
            return 0.0
        current_year = datetime.utcnow().year
        total_years = 0.0
        for item in experience:
            if not isinstance(item, dict):
                continue
            start = self._parse_year(item.get("start"))
            end = self._parse_year(item.get("end"))
            if start is None:
                continue
            if end is None:
                end_text = self._to_text(item.get("end")).lower()
                end = current_year if end_text in {"present", "actual", "current", ""} else start
            if end >= start:
                total_years += float(end - start)
        return total_years

    def _technology_match_score(
        self,
        candidate_profile: dict,
        vacancy_profile: dict,
    ) -> tuple[float, dict]:
        candidate_tech = self._extract_candidate_technologies(candidate_profile)
        vacancy_tech = self._extract_vacancy_technologies(vacancy_profile)
        matched = sorted(candidate_tech.intersection(vacancy_tech))
        missing = sorted(vacancy_tech.difference(candidate_tech))

        if not vacancy_tech:
            return 60.0, {"matched": matched, "missing": missing}

        coverage = len(matched) / len(vacancy_tech)
        score = coverage * 100.0
        return round(score, 2), {"matched": matched, "missing": missing}

    def _requirements_match_score(
        self,
        candidate_profile: dict,
        vacancy_profile: dict,
    ) -> tuple[float, dict]:
        requirement_keywords = self._extract_requirement_keywords(vacancy_profile)
        candidate_blob = self._candidate_context_blob(candidate_profile)
        soft_blob = self._candidate_soft_skills_blob(candidate_profile)

        hard_requirements = requirement_keywords["hard"]
        soft_requirements = requirement_keywords["soft"]

        matched_hard = sorted(
            keyword
            for keyword in hard_requirements
            if self._candidate_matches_keyword(candidate_blob, keyword)
        )
        matched_soft = sorted(
            keyword
            for keyword in soft_requirements
            if self._candidate_matches_soft_skill(soft_blob, keyword)
        )

        if not hard_requirements and not soft_requirements:
            return 60.0, {"matched": [], "missing": [], "hard": {}, "soft": {}}

        hard_score = (
            (len(matched_hard) / len(hard_requirements)) * 100.0 if hard_requirements else 60.0
        )
        soft_score = (
            (len(matched_soft) / len(soft_requirements)) * 100.0 if soft_requirements else 60.0
        )
        score = (hard_score * 0.8) + (soft_score * 0.2)

        missing_hard = sorted(hard_requirements.difference(matched_hard))
        missing_soft = sorted(soft_requirements.difference(matched_soft))
        return round(score, 2), {
            "matched": sorted(matched_hard + matched_soft),
            "missing": sorted(missing_hard + missing_soft),
            "hard": {"matched": matched_hard, "missing": missing_hard},
            "soft": {"matched": matched_soft, "missing": missing_soft},
        }

    def _context_score(
        self,
        candidate_profile: dict,
        vacancy_profile: dict,
    ) -> tuple[float, list[str]]:
        score = 60.0
        notes: list[str] = []

        candidate_location = self._normalize_token(self._to_text(candidate_profile.get("location")))
        vacancy_location = self._normalize_token(self._to_text(vacancy_profile.get("location")))
        modality = self._normalize_token(self._to_text(vacancy_profile.get("modality")))

        if vacancy_location and candidate_location:
            if candidate_location == vacancy_location:
                score += 15.0
                notes.append("ubicacion alineada")
            elif modality == "remote":
                score += 5.0
            else:
                score -= 10.0
                notes.append("ubicacion potencialmente incompatible")

        expected_salary = candidate_profile.get("expected_salary")
        min_salary = vacancy_profile.get("min_salary")
        max_salary = vacancy_profile.get("max_salary")
        if isinstance(expected_salary, (int, float)) and isinstance(max_salary, (int, float)):
            if min_salary is not None and min_salary <= expected_salary <= max_salary:
                score += 15.0
                notes.append("expectativa salarial dentro del rango")
            elif expected_salary <= max_salary * 1.1:
                score += 5.0
                notes.append("expectativa salarial cercana al rango")
            else:
                score -= 15.0
                notes.append("expectativa salarial por encima del rango")

        candidate_languages = self._extract_languages(candidate_profile.get("languages"))
        required_languages = self._extract_required_languages(vacancy_profile)
        if required_languages:
            matched_languages = candidate_languages.intersection(required_languages)
            if matched_languages == required_languages:
                score += 10.0
                notes.append("idiomas requeridos cubiertos")
            elif matched_languages:
                score += 4.0
                notes.append("idiomas cubiertos parcialmente")
            else:
                score -= 8.0
                notes.append("faltan idiomas requeridos")

        return round(self._clamp(score), 2), notes

    def _cap_semantic_score(self, semantic_score: float, technology_score: float) -> float:
        if technology_score < 20:
            return min(semantic_score, 35.0)
        if technology_score < 40:
            return min(semantic_score, 55.0)
        if technology_score < 60:
            return min(semantic_score, 75.0)
        return semantic_score

    def _normalize_dimension_score(self, value, fallback: float = 50.0) -> float:
        if isinstance(value, (int, float)):
            return round(self._clamp(float(value)), 2)
        return round(self._clamp(fallback), 2)

    def _normalize_llm_dimension(self, llm_evaluation: dict, key: str, fallback: dict) -> dict:
        dimension = llm_evaluation.get(key)
        if not isinstance(dimension, dict):
            return fallback
        return {
            "score": self._normalize_dimension_score(dimension.get("score"), fallback["score"]),
            "matched": self._value_to_list(dimension.get("matched")) or fallback.get("matched", []),
            "missing": self._value_to_list(dimension.get("missing")) or fallback.get("missing", []),
            "rationale": self._to_text(dimension.get("rationale")) or fallback.get("rationale", ""),
        }

    def _normalize_red_flag_type(self, red_flag_type: str) -> str:
        normalized = self._normalize_token(red_flag_type.replace("_", " "))
        aliases = {
            "missing_technology": "missing_technologies",
            "missing technology": "missing_technologies",
            "missing_technologies": "missing_technologies",
            "missing technologies": "missing_technologies",
            "experience_gap": "experience_level_gap",
            "experience gap": "experience_level_gap",
            "experience_level_gap": "experience_level_gap",
            "experience level gap": "experience_level_gap",
            "salary_gap": "salary_gap",
            "salary gap": "salary_gap",
            "location_gap": "location_gap",
            "location gap": "location_gap",
        }
        return aliases.get(normalized, normalized)

    def _red_flag_penalty(self, red_flags, hard_penalties=None) -> tuple[float, list[str]]:
        severity_penalties = {"low": 2.0, "medium": 5.0, "high": 10.0}
        penalty = 0.0
        reasons: list[str] = []
        covered_types = {
            self._normalize_red_flag_type(self._to_text(item.get("type")))
            for item in (hard_penalties or [])
            if isinstance(item, dict)
        }
        if not isinstance(red_flags, list):
            return penalty, reasons
        for item in red_flags:
            if not isinstance(item, dict):
                continue
            normalized_type = self._normalize_red_flag_type(self._to_text(item.get("type")))
            if normalized_type in covered_types:
                continue
            severity = self._to_text(item.get("severity")).lower() or "low"
            detail = self._to_text(item.get("detail"))
            penalty += severity_penalties.get(severity, 2.0)
            if detail:
                reasons.append(detail)
        return round(penalty, 2), reasons

    def validate_hard_requirements(
        self,
        candidate_profile: dict,
        vacancy_profile: dict,
    ) -> dict:
        penalties: list[dict] = []
        reasons: list[str] = []
        total_penalty = 0.0

        tech_score, tech_details = self._technology_match_score(candidate_profile, vacancy_profile)
        missing_tech = tech_details["missing"]
        total_required_tech = len(missing_tech) + len(tech_details["matched"])
        if total_required_tech > 0 and missing_tech:
            missing_ratio = len(missing_tech) / total_required_tech
            if missing_ratio >= 0.75:
                penalty = 30.0
            elif missing_ratio >= 0.5:
                penalty = 20.0
            else:
                penalty = 10.0
            penalties.append(
                {
                    "type": "missing_technologies",
                    "penalty": penalty,
                    "details": missing_tech,
                }
            )
            reasons.append(
                "Faltan tecnologias clave: " + ", ".join(missing_tech[:4])
            )
            total_penalty += penalty

        candidate_level = self._infer_experience_level(candidate_profile)
        required_level = self._required_experience_level(vacancy_profile)
        if candidate_level < required_level:
            penalty = float((required_level - candidate_level) * 8)
            penalties.append(
                {
                    "type": "experience_level_gap",
                    "penalty": penalty,
                    "details": {
                        "candidate_level": candidate_level,
                        "required_level": required_level,
                    },
                }
            )
            reasons.append("El seniority del candidato parece inferior al requerido")
            total_penalty += penalty

        expected_salary = candidate_profile.get("expected_salary")
        max_salary = vacancy_profile.get("max_salary")
        if isinstance(expected_salary, (int, float)) and isinstance(max_salary, (int, float)):
            if expected_salary > max_salary * 1.2:
                penalty = 15.0
                penalties.append(
                    {
                        "type": "salary_gap",
                        "penalty": penalty,
                        "details": {
                            "expected_salary": expected_salary,
                            "max_salary": max_salary,
                        },
                    }
                )
                reasons.append("La expectativa salarial supera claramente el rango")
                total_penalty += penalty

        candidate_location = self._normalize_token(self._to_text(candidate_profile.get("location")))
        vacancy_location = self._normalize_token(self._to_text(vacancy_profile.get("location")))
        modality = self._normalize_token(self._to_text(vacancy_profile.get("modality")))
        if (
            modality in {"on site", "onsite", "hybrid"}
            and candidate_location
            and vacancy_location
            and candidate_location != vacancy_location
        ):
            penalty = 8.0
            penalties.append(
                {
                    "type": "location_gap",
                    "penalty": penalty,
                    "details": {
                        "candidate_location": candidate_location,
                        "vacancy_location": vacancy_location,
                        "modality": modality,
                    },
                }
            )
            reasons.append("La ubicacion no coincide con una modalidad presencial/hibrida")
            total_penalty += penalty

        return {
            "passed": total_penalty == 0.0,
            "penalties": penalties,
            "penalty_points": round(total_penalty, 2),
            "reasons": reasons,
            "tech_score": tech_score,
        }

    def score_structured_match(
        self,
        candidate_profile: dict,
        vacancy_profile: dict,
        similarity_score: float,
        llm_evaluation: dict | None = None,
    ) -> dict:
        semantic_score = self._normalize_similarity_score(similarity_score)
        deterministic_technology_score, technology_details = self._technology_match_score(
            candidate_profile,
            vacancy_profile,
        )
        deterministic_requirements_score, requirements_details = self._requirements_match_score(
            candidate_profile,
            vacancy_profile,
        )
        deterministic_context_score, context_notes = self._context_score(
            candidate_profile,
            vacancy_profile,
        )
        hard_requirements = self.validate_hard_requirements(candidate_profile, vacancy_profile)
        semantic_score = self._cap_semantic_score(semantic_score, deterministic_technology_score)

        experience_level_gap = next(
            (
                item["penalty"]
                for item in hard_requirements["penalties"]
                if item.get("type") == "experience_level_gap"
            ),
            0.0,
        )
        deterministic_experience_score = round(self._clamp(80.0 - experience_level_gap * 4), 2)

        technology_dimension = {
            "score": deterministic_technology_score,
            "matched": technology_details["matched"],
            "missing": technology_details["missing"],
            "rationale": "",
        }
        experience_dimension = {
            "score": deterministic_experience_score,
            "matched": [],
            "missing": [],
            "rationale": "",
        }
        requirements_dimension = {
            "score": deterministic_requirements_score,
            "matched": requirements_details["matched"],
            "missing": requirements_details["missing"],
            "rationale": "",
        }
        context_dimension = {
            "score": deterministic_context_score,
            "matched": context_notes,
            "missing": [],
            "rationale": "",
        }

        llm_red_flags: list = []
        llm_summary = ""
        llm_evaluation_used = isinstance(llm_evaluation, dict)
        if isinstance(llm_evaluation, dict):
            technology_dimension = self._normalize_llm_dimension(
                llm_evaluation,
                "technology_fit",
                technology_dimension,
            )
            experience_dimension = self._normalize_llm_dimension(
                llm_evaluation,
                "experience_fit",
                experience_dimension,
            )
            requirements_dimension = self._normalize_llm_dimension(
                llm_evaluation,
                "requirements_fit",
                requirements_dimension,
            )
            context_dimension = self._normalize_llm_dimension(
                llm_evaluation,
                "context_fit",
                context_dimension,
            )
            llm_red_flags = llm_evaluation.get("red_flags", []) if isinstance(
                llm_evaluation.get("red_flags"), list
            ) else []
            llm_summary = self._to_text(llm_evaluation.get("summary"))

        red_flag_penalty, red_flag_reasons = self._red_flag_penalty(
            llm_red_flags,
            hard_requirements["penalties"],
        )

        blended_score = (
            (technology_dimension["score"] * 0.35)
            + (experience_dimension["score"] * 0.25)
            + (requirements_dimension["score"] * 0.25)
            + (context_dimension["score"] * 0.15)
        )
        final_score = round(
            self._clamp(blended_score - hard_requirements["penalty_points"] - red_flag_penalty),
            2,
        )

        reasons = []
        if technology_dimension["matched"]:
            reasons.append(
                "Tecnologias alineadas: " + ", ".join(technology_dimension["matched"][:4])
            )
        reasons.extend(hard_requirements["reasons"])
        reasons.extend(red_flag_reasons)
        reasons.extend(context_notes)
        if llm_summary:
            reasons.append(llm_summary)

        return {
            "semantic_score": semantic_score,
            "technology_score": technology_dimension["score"],
            "experience_score": experience_dimension["score"],
            "requirements_score": requirements_dimension["score"],
            "context_score": context_dimension["score"],
            "compatibility_percentage": final_score,
            "hard_requirements": hard_requirements,
            "reasons": reasons[:6],
            "details": {
                "technology": {
                    "matched": technology_dimension["matched"],
                    "missing": technology_dimension["missing"],
                    "rationale": technology_dimension["rationale"],
                },
                "experience": {
                    "matched": experience_dimension["matched"],
                    "missing": experience_dimension["missing"],
                    "rationale": experience_dimension["rationale"],
                },
                "requirements": {
                    "matched": requirements_dimension["matched"],
                    "missing": requirements_dimension["missing"],
                    "rationale": requirements_dimension["rationale"],
                    "source": "llm" if llm_evaluation_used else "fallback",
                    "hard": {
                        "matched": requirements_dimension["matched"],
                        "missing": requirements_dimension["missing"],
                    } if llm_evaluation_used else requirements_details.get("hard", {}),
                    "soft": {
                        "matched": [],
                        "missing": [],
                    } if llm_evaluation_used else requirements_details.get("soft", {}),
                    "fallback_assessment": {
                        "matched": requirements_details.get("matched", []),
                        "missing": requirements_details.get("missing", []),
                        "hard": requirements_details.get("hard", {}),
                        "soft": requirements_details.get("soft", {}),
                    } if llm_evaluation_used else None,
                },
                "context": {
                    "matched": context_dimension["matched"],
                    "missing": context_dimension["missing"],
                    "rationale": context_dimension["rationale"],
                },
            },
            "matched_technologies": technology_dimension["matched"],
            "missing_technologies": technology_dimension["missing"],
            "hard_requirement_failures": hard_requirements["penalties"],
            "soft_matches": (
                []
                if llm_evaluation_used
                else requirements_details.get("soft", {}).get("matched", [])
            ),
            "llm_evaluation_used": llm_evaluation_used,
            "llm_red_flags": llm_red_flags,
            "llm_red_flag_penalty": red_flag_penalty,
        }

    def to_compatibility_percentage(self, similarity_score: float) -> float:
        return self._normalize_similarity_score(similarity_score)

    def compatibility_level(self, compatibility_percentage: float) -> str:
        if compatibility_percentage >= 75:
            return "high"
        if compatibility_percentage >= 50:
            return "medium"
        return "low"

    def generate_rule_based_feedback(
        self,
        compatibility_percentage: float,
        reasons: list[str] | None = None,
    ) -> str:
        level = self.compatibility_level(compatibility_percentage)
        reasons = [reason for reason in (reasons or []) if reason]
        reasons_text = ""
        if reasons:
            reasons_text = " Hallazgos principales: " + "; ".join(reasons[:3]) + "."

        if level == "high":
            return (
                "El perfil del candidato muestra una compatibilidad alta considerando "
                "similitud semantica, stack y contexto del rol."
                + reasons_text
            )

        if level == "medium":
            return (
                "Existe compatibilidad parcial entre candidato y vacante. Conviene validar "
                "brechas tecnicas o de contexto antes de avanzar."
                + reasons_text
            )

        return (
            "La compatibilidad es baja con base en requisitos clave, stack y contexto del rol."
            + reasons_text
        )
