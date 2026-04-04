import numpy as np


class MatchingService:

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

        similarity = dot_product / (norm_a * norm_b)

        return float(similarity)

    def to_compatibility_percentage(self, similarity_score: float) -> float:
        # Normalize cosine similarity from [-1, 1] to [0, 100].
        normalized = ((similarity_score + 1.0) / 2.0) * 100.0
        clamped = max(0.0, min(100.0, normalized))
        return round(clamped, 2)

    def compatibility_level(self, compatibility_percentage: float) -> str:
        if compatibility_percentage >= 80:
            return "high"
        if compatibility_percentage >= 60:
            return "medium"
        return "low"

    def generate_rule_based_feedback(self, compatibility_percentage: float) -> str:
        level = self.compatibility_level(compatibility_percentage)

        if level == "high":
            return (
                "El perfil del candidato parece alineado con la vacante en experiencia, "
                "habilidades y contexto general. Recomendado para avanzar a una entrevista "
                "técnica/cultural y validar expectativas salariales."
            )

        if level == "medium":
            return (
                "Existe compatibilidad parcial entre candidato y vacante. Se recomienda "
                "validar brechas puntuales de stack, nivel de seniority o experiencia de dominio "
                "antes de decidir el siguiente paso."
            )

        return (
            "La compatibilidad es baja con base en la información textual comparada. "
            "Podrían existir diferencias importantes en habilidades clave, seniority o enfoque "
            "del rol."
        )