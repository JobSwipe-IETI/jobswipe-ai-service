import json
import os
import re

from openai import AzureOpenAI, OpenAI

from app.models.schemas import CandidateProfileInput
import uuid


class AzureLLMService:
    ALLOWED_SECTORS = [
        "Tecnologia",
        "Finanzas",
        "Salud",
        "Educacion",
        "Retail",
        "Logistica",
        "Marketing",
        "Construccion",
        "Energia",
        "Servicios",
        "Telecomunicaciones",
        "Otro",
    ]
    PROFILE_SECTION_SKILLS = "Habilidades"
    PROFILE_SECTION_GENERAL = "Perfil general"
    PROFILE_SECTION_EXPERIENCE = "Experiencia"
    PROFILE_SECTION_EDUCATION = "Educación"
    PROFILE_SECTION_ADDITIONAL = "Información adicional"

    def __init__(self):
        # Dedicated vars for PDF/chat model, with fallback to shared Azure OpenAI vars.
        self.api_key = os.getenv(
            "AZURE_OPENAI_PDF_API_KEY",
            os.getenv("AZURE_OPENAI_API_KEY"),
        )
        self.endpoint = os.getenv(
            "AZURE_OPENAI_PDF_ENDPOINT",
            os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.chat_deployment = os.getenv(
            "AZURE_OPENAI_PDF_DEPLOYMENT",
            os.getenv(
                "AZURE_OPENAI_CHAT_DEPLOYMENT",
                os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            ),
        )
        self.api_version = os.getenv(
            "AZURE_OPENAI_PDF_API_VERSION",
            os.getenv(
                "AZURE_OPENAI_CHAT_API_VERSION",
                os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            ),
        )

        self._client = None
        self._use_project_responses_api = False
        if self.is_configured():
            # Azure AI Foundry project endpoints work best with OpenAI(base_url=...)
            # and the Responses API.
            if "/api/projects/" in (self.endpoint or ""):
                self._use_project_responses_api = True
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.endpoint.rstrip("/"),
                )
            else:
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version,
                )

    def _normalize_text(self, value):
        return (value or "").strip()

    def _normalize_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    def _has_value(self, value):
        if value is None:
            return False
        if isinstance(value, list):
            return any(self._normalize_text(item) for item in value)
        return bool(self._normalize_text(value))

    def _create_profile_suggestion(self, section, msg, severity="medium", example=None):
        suggestion = {
            "id": f"static-{uuid.uuid4().hex[:8]}",
            "category": section,
            "message": msg,
            "severity": severity,
        }
        if example:
            suggestion["example"] = example
        return suggestion

    def _profile_section_score(self, items: list[dict]) -> int:
        high = sum(1 for item in items if item.get("severity") == "high")
        medium = sum(1 for item in items if item.get("severity") == "medium")
        score = 95 - high * 20 - medium * 8
        return max(20, min(100, score))

    def _build_profile_sections(self, suggestions: list[dict]):
        section_order = [self.PROFILE_SECTION_GENERAL, self.PROFILE_SECTION_EXPERIENCE, self.PROFILE_SECTION_EDUCATION, self.PROFILE_SECTION_ADDITIONAL]
        grouped: dict[str, list[dict]] = {}
        for item in suggestions:
            grouped.setdefault(item.get("category") or self.PROFILE_SECTION_ADDITIONAL, []).append(item)

        final_sections = []
        for name in section_order:
            section_items = grouped.get(name, [])
            if not section_items:
                continue
            final_sections.append({
                "name": name,
                "score": self._profile_section_score(section_items),
                "summary": section_items[0].get("message", "") if section_items else "",
                "suggestions": section_items,
            })

        for name, section_items in grouped.items():
            if name in section_order:
                continue
            final_sections.append({
                "name": name,
                "score": self._profile_section_score(section_items),
                "summary": section_items[0].get("message", "") if section_items else "",
                "suggestions": section_items,
            })

        return final_sections

    def _build_general_profile_suggestions(self, candidate_profile: dict):
        suggestions = []
        prof_title = self._normalize_text(candidate_profile.get("professionalTitle"))
        summary = self._normalize_text(candidate_profile.get("summary"))

        if not prof_title:
            suggestions.append(self._create_profile_suggestion(
                self.PROFILE_SECTION_ADDITIONAL,
                "Falta el título profesional. Usa un cargo claro y específico para que el perfil se entienda rápido.",
                "high",
                "Ejemplo: Backend Engineer | Java | Spring Boot | PostgreSQL",
            ))

        if not summary or len(summary) < 40:
            suggestions.append(self._create_profile_suggestion(
                self.PROFILE_SECTION_ADDITIONAL,
                "El resumen es muy corto o está vacío. Resume tu propuesta de valor en 2 a 3 líneas con foco en logros, stack y tipo de rol.",
                "high",
                "Ejemplo: Desarrollador backend con experiencia en APIs REST, automatización y despliegue en la nube.",
            ))

        return suggestions

    def _extract_experience_techs(self, experience_items):
        techs = []
        for item in experience_items:
            if not isinstance(item, dict):
                continue
            value = item.get("tech") or item.get("technologies") or []
            for tech in self._normalize_list(value):
                normalized = self._normalize_text(tech)
                if normalized and normalized.lower() not in {existing.lower() for existing in techs}:
                    techs.append(normalized)
        return techs

    def _build_skills_profile_suggestions(self, candidate_profile: dict):
        suggestions = []
        skills = self._normalize_list(candidate_profile.get("skills"))
        experience = candidate_profile.get("experience") or []

        experience_techs = self._extract_experience_techs(experience if isinstance(experience, list) else [])
        skill_set = {self._normalize_text(skill).lower() for skill in skills if self._normalize_text(skill)}

        missing_techs = [tech for tech in experience_techs if tech.lower() not in skill_set]
        if missing_techs:
            suggestions.append(self._create_profile_suggestion(
                self.PROFILE_SECTION_SKILLS,
                f"Tienes experiencia con estas tecnologías, pero no aparecen en skills: {', '.join(missing_techs[:5])}. Agrégalas si realmente las dominas.",
                "high",
                f"Ejemplo: añade {missing_techs[0]} si también la usaste en tus proyectos o experiencia.",
            ))

        generic_skills = [
            skill for skill in skills
            if self._normalize_text(skill).lower() in {
                'trabajo en equipo', 'responsable', 'proactivo', 'comunicacion',
                'liderazgo', 'adaptabilidad', 'aprendizaje rapido', 'autodidacta',
            }
        ]
        if generic_skills:
            suggestions.append(self._create_profile_suggestion(
                self.PROFILE_SECTION_SKILLS,
                f"Hay skills muy genéricas que conviene reemplazar por tecnologías o herramientas concretas: {', '.join(generic_skills[:4])}.",
                "medium",
                "Ejemplo: deja las habilidades blandas para otra parte del CV y aquí prioriza stack técnico.",
            ))

        if not skills:
            suggestions.append(self._create_profile_suggestion(
                self.PROFILE_SECTION_SKILLS,
                "No hay habilidades registradas. Agrega tecnologías, herramientas o frameworks que ya hayas usado en experiencia o proyectos.",
                "high",
                "Ejemplo: Java, Spring Boot, PostgreSQL, Docker, Azure, Git",
            ))

        return suggestions

    def _build_experience_profile_suggestions(self, candidate_profile: dict):
        suggestions = []
        experience = self._normalize_list(candidate_profile.get("experience"))

        if not experience:
            suggestions.append(self._create_profile_suggestion(
                self.PROFILE_SECTION_EXPERIENCE,
                "No hay experiencia laboral registrada. Agrega al menos un rol con funciones, resultados y tecnologías usadas.",
                "high",
                "Ejemplo: Backend Developer - Construí APIs REST y optimicé consultas SQL.",
            ))
            return suggestions

        first_experience = experience[0]
        if isinstance(first_experience, dict):
            missing_experience_bits = []
            if not self._has_value(first_experience.get("role")):
                missing_experience_bits.append("rol")
            if not self._has_value(first_experience.get("description")):
                missing_experience_bits.append("descripción")
            if not self._has_value(first_experience.get("tech")):
                missing_experience_bits.append("tecnologías")
            if missing_experience_bits:
                suggestions.append(self._create_profile_suggestion(
                    self.PROFILE_SECTION_EXPERIENCE,
                    f"Tu experiencia inicial podría ser más sólida: faltan {', '.join(missing_experience_bits)}. Explica qué hiciste, con qué tecnologías y qué lograste.",
                    "medium",
                    "Ejemplo: agrega funciones concretas, stack y resultados medibles.",
                ))

        return suggestions

    def _build_education_profile_suggestions(self, candidate_profile: dict):
        suggestions = []
        education = self._normalize_list(candidate_profile.get("education"))

        if not education:
            suggestions.append(self._create_profile_suggestion(
                self.PROFILE_SECTION_EDUCATION,
                "No aparece formación académica o certificaciones. Añade estudios, estado y fechas para dar contexto al perfil.",
                "medium",
                "Ejemplo: Ingeniería de Sistemas - Universidad X (en curso o graduado).",
            ))
            return suggestions

        first_education = education[0]
        if isinstance(first_education, dict):
            missing_education_bits = []
            if not self._has_value(first_education.get("institution")):
                missing_education_bits.append("institución")
            if not self._has_value(first_education.get("degree")):
                missing_education_bits.append("programa o grado")
            if missing_education_bits:
                suggestions.append(self._create_profile_suggestion(
                    self.PROFILE_SECTION_EDUCATION,
                    f"Tu formación podría detallarse mejor: faltan {', '.join(missing_education_bits)}.",
                    "medium",
                ))

        return suggestions

    def _build_additional_profile_suggestions(self, candidate_profile: dict):
        suggestions = []
        location = self._normalize_text(candidate_profile.get("location"))
        nationality = self._normalize_text(candidate_profile.get("nationality"))
        languages = self._normalize_list(candidate_profile.get("languages"))
        expected_salary = candidate_profile.get("expectedSalary")
        availability = self._normalize_text(candidate_profile.get("availability"))
        email = self._normalize_text(candidate_profile.get("email"))
        phone = self._normalize_text(candidate_profile.get("phoneNumber"))
        github = self._normalize_list(candidate_profile.get("github"))
        linkedin = self._normalize_list(candidate_profile.get("linkedin"))
        links = self._normalize_list(candidate_profile.get("links"))

        additional_gaps = []
        if not location:
            additional_gaps.append("ubicación")
        if not nationality:
            additional_gaps.append("nacionalidad")
        if not languages:
            additional_gaps.append("idiomas")
        if expected_salary is None:
            additional_gaps.append("salario esperado")
        if not availability:
            additional_gaps.append("disponibilidad")
        if not email and not phone:
            additional_gaps.append("contacto")

        if additional_gaps:
            suggestions.append(self._create_profile_suggestion(
                self.PROFILE_SECTION_ADDITIONAL,
                f"Faltan datos clave de contexto: {', '.join(additional_gaps)}. Completa esa sección para que el perfil se vea más completo.",
                "medium",
                "Ejemplo: ciudad, nacionalidad, idiomas, salario esperado, disponibilidad y contacto.",
            ))

        if not github and not linkedin and not links:
            suggestions.append(self._create_profile_suggestion(
                self.PROFILE_SECTION_ADDITIONAL,
                "No hay enlaces profesionales. Agrega GitHub, LinkedIn o portafolio para reforzar credibilidad.",
                "medium",
                "Ejemplo: GitHub + LinkedIn + portafolio personal.",
            ))

        return suggestions

    def _build_static_profile_feedback(self, candidate_profile: dict):
        suggestions: list[dict] = []
        suggestions.extend(self._build_skills_profile_suggestions(candidate_profile))
        suggestions.extend(self._build_general_profile_suggestions(candidate_profile))
        suggestions.extend(self._build_experience_profile_suggestions(candidate_profile))
        suggestions.extend(self._build_education_profile_suggestions(candidate_profile))
        suggestions.extend(self._build_additional_profile_suggestions(candidate_profile))

        missing_criticals = sum(1 for item in suggestions if item["severity"] == "high")
        missing_mediums = sum(1 for item in suggestions if item["severity"] == "medium")
        overall_score = max(20, 95 - missing_criticals * 18 - missing_mediums * 8)

        if overall_score >= 85:
            short_summary = "Buen perfil: tienes una base sólida y solo faltan detalles puntuales."
        elif overall_score >= 65:
            short_summary = "Buen perfil con oportunidades claras de mejora por secciones."
        else:
            short_summary = "Perfil con varias áreas por reforzar para verse más competitivo."

        return suggestions, overall_score, short_summary

    def _build_profile_prompts(self, candidate_profile: dict):
        system_prompt = (
            "Eres un asesor de mejora de perfiles profesionales en español. "
            "Devuelve SOLO JSON valido, sin markdown ni texto adicional. "
            "Organiza el feedback en secciones claras y consistentes: Habilidades, Experiencia, Educación e Información adicional. "
            "Cada seccion debe incluir: name, score (0-100), summary y suggestions. "
            "Cada sugerencia debe ser concreta, accionable y especifica, y debe corresponder a campos que la app realmente permite editar. "
            "Reglas por seccion: "
            "Habilidades: solo sugiere agregar habilidades que ya aparezcan en experiencia/proyectos o quitar skills demasiado genericas; no hables de resumen, experiencia o educacion. "
            "Experiencia: usa role, project, company, tech, description, start y end; puedes pedir responsabilidades, tecnologias y resultados cuantificables porque description lo permite. "
            "Educación: usa institution, degree, start, end y status; si la persona estudia actualmente, sugiere 'en curso' y/o completar fechas, no pidas campos inexistentes. "
            "Información adicional: usa summary, location, nationality, languages, expectedSalary, availability, email, phoneNumber, github, linkedin y links; no menciones presencial/on-site para availability, porque availability no representa modalidad laboral. "
            "Evita recomendaciones genericas como 'mejorar el perfil' o 'agregar mas detalles'."
        )

        user_prompt = (
            "Analiza este perfil y devuelve el JSON pedido. Asegurate de no incluir texto adicional. "
            "Estructura la respuesta asi:\n"
            "{\n"
            '  "overall_score": 0,\n'
            '  "sections": [\n'
            '    {"name": "Habilidades", "score": 0, "summary": "...", "suggestions": [{"id": "s1", "message": "...", "severity": "low|medium|high", "example": "..."}]},\n'
            '    {"name": "Experiencia", "score": 0, "summary": "...", "suggestions": [{"id": "s2", "message": "...", "severity": "low|medium|high", "example": "..."}]},\n'
            '    {"name": "Educación", "score": 0, "summary": "...", "suggestions": [{"id": "s3", "message": "...", "severity": "low|medium|high", "example": "..."}]},\n'
            '    {"name": "Información adicional", "score": 0, "summary": "...", "suggestions": [{"id": "s4", "message": "...", "severity": "low|medium|high", "example": "..."}]}\n'
            "  ],\n"
            '  "metadata": {"model": "..."}\n'
            "}\n\n"
            f"candidateProfile:\n{json.dumps(candidate_profile, ensure_ascii=False)}"
        )

        return system_prompt, user_prompt

    def _collect_ai_section_suggestions(self, ai_result: dict):
        suggestions = []
        for section in ai_result.get("sections") or []:
            if not isinstance(section, dict):
                continue
            section_name = section.get("name") or section.get("category") or self.PROFILE_SECTION_ADDITIONAL
            for item in section.get("suggestions") or section.get("items") or []:
                if not isinstance(item, dict):
                    continue
                normalized_item = dict(item)
                normalized_item.setdefault("id", f"ai-{uuid.uuid4().hex[:8]}")
                normalized_item["category"] = section_name
                suggestions.append(normalized_item)
        return suggestions

    def _collect_ai_flat_suggestions(self, ai_result: dict):
        suggestions = []
        for item in ai_result.get("suggestions") or ai_result.get("recommendations") or []:
            if not isinstance(item, dict):
                continue
            normalized_item = dict(item)
            normalized_item.setdefault("id", f"ai-{uuid.uuid4().hex[:8]}")
            suggestions.append(normalized_item)
        return suggestions

    def _collect_ai_profile_suggestions(self, ai_result: dict):
        if not isinstance(ai_result, dict):
            return []
        return self._collect_ai_section_suggestions(ai_result) + self._collect_ai_flat_suggestions(ai_result)

    def _get_ai_profile_feedback(self, candidate_profile: dict):
        ai_result = {}
        if not self._client:
            return ai_result

        system_prompt, user_prompt = self._build_profile_prompts(candidate_profile)
        try:
            if self._use_project_responses_api:
                response = self._client.responses.create(
                    model=self.chat_deployment,
                    instructions=system_prompt,
                    input=user_prompt,
                    max_output_tokens=800,
                )
                content = getattr(response, "output_text", "{}") or "{}"
                ai_result = self._safe_json_loads(content)
            else:
                completion = self._client.chat.completions.create(
                    model=self.chat_deployment,
                    temperature=0.1,
                    max_tokens=800,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                if completion.choices:
                    content = completion.choices[0].message.content or "{}"
                    ai_result = self._safe_json_loads(content)
        except Exception:
            ai_result = {}

        return ai_result

    def _merge_profile_feedback(self, static_suggestions: list[dict], ai_result: dict, overall_score: int, short_summary: str):
        final_suggestions = list(static_suggestions)

        if isinstance(ai_result, dict):
            final_suggestions.extend(self._collect_ai_profile_suggestions(ai_result))

            if isinstance(ai_result.get("overall_score"), (int, float)):
                overall_score = int(ai_result.get("overall_score"))

            if isinstance(ai_result.get("summary"), str) and ai_result.get("summary").strip():
                short_summary = ai_result.get("summary").strip()

        final_sections = self._build_profile_sections(final_suggestions)

        return {
            "overall_score": overall_score,
            "summary": short_summary,
            "sections": final_sections,
            "categories": ai_result.get("categories") if isinstance(ai_result, dict) else [],
            "suggestions": final_suggestions,
            "used_ai": bool(self._client),
            "metadata": ai_result.get("metadata") if isinstance(ai_result, dict) else {},
        }

    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint and self.chat_deployment)

    def generate_match_feedback(
        self,
        candidate_text: str,
        vacancy_text: str,
        compatibility_percentage: float,
    ) -> str | None:
        if not self._client:
            return None

        system_prompt = (
            "Eres un experto reclutador tecnico. Devuelve feedback conciso en espanol "
            "con: resumen de compatibilidad, fortalezas, brechas y recomendacion. "
            "Maximo 5 lineas, accionable y claro."
        )

        user_prompt = (
            f"Compatibilidad estimada por embeddings: {compatibility_percentage}%.\n\n"
            f"Candidato:\n{candidate_text}\n\n"
            f"Vacante:\n{vacancy_text}\n\n"
            "Analiza y devuelve feedback corto."
        )

        try:
            if self._use_project_responses_api:
                response = self._client.responses.create(
                    model=self.chat_deployment,
                    instructions=system_prompt,
                    input=user_prompt,
                    max_output_tokens=250,
                )
                content = getattr(response, "output_text", "") or ""
                return content.strip() or None

            completion = self._client.chat.completions.create(
                model=self.chat_deployment,
                temperature=0.2,
                max_tokens=250,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            if not completion.choices:
                return None

            content = completion.choices[0].message.content or ""
            return content.strip() or None
        except Exception:
            return None

    def evaluate_match_dimensions(
        self,
        candidate_profile: dict,
        vacancy_profile: dict,
    ) -> dict | None:
        if not self._client:
            return None

        system_prompt = (
            "Eres un evaluador tecnico de matching laboral. "
            "Analiza un candidateProfile y un vacancyProfile y devuelve SOLO JSON valido. "
            "No devuelvas markdown, explicaciones ni texto adicional. "
            "Debes evaluar de 0 a 100 estas dimensiones: technology_fit, experience_fit, "
            "requirements_fit, context_fit. "
            "Cada dimension debe incluir: score, matched, missing, rationale. "
            "Adicionalmente devuelve red_flags como lista de objetos con keys: type, severity, detail. "
            "Usa severity en low, medium o high. "
            "No inventes experiencia si no hay evidencia en el perfil. "
            "Considera technologies, skills, summary, experience, education, languages, sector, location, salary y modality."
        )

        user_prompt = (
            "Evalua la compatibilidad entre este perfil y esta vacante. "
            "Responde exclusivamente con JSON valido usando esta estructura:\n"
            "{\n"
            '  "technology_fit": {"score": 0, "matched": [], "missing": [], "rationale": ""},\n'
            '  "experience_fit": {"score": 0, "matched": [], "missing": [], "rationale": ""},\n'
            '  "requirements_fit": {"score": 0, "matched": [], "missing": [], "rationale": ""},\n'
            '  "context_fit": {"score": 0, "matched": [], "missing": [], "rationale": ""},\n'
            '  "red_flags": [{"type": "missing_technology", "severity": "high", "detail": "Falta FastAPI"}],\n'
            '  "summary": "Resumen corto"\n'
            "}\n\n"
            f"candidateProfile:\n{json.dumps(candidate_profile, ensure_ascii=False)}\n\n"
            f"vacancyProfile:\n{json.dumps(vacancy_profile, ensure_ascii=False)}"
        )

        try:
            if self._use_project_responses_api:
                response = self._client.responses.create(
                    model=self.chat_deployment,
                    instructions=system_prompt,
                    input=user_prompt,
                    max_output_tokens=1200,
                )
                content = getattr(response, "output_text", "{}") or "{}"
                return self._safe_json_loads(content)

            completion = self._client.chat.completions.create(
                model=self.chat_deployment,
                temperature=0.1,
                max_tokens=1200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            if not completion.choices:
                return None

            content = completion.choices[0].message.content or "{}"
            data = self._safe_json_loads(content)
            return data or None
        except Exception:
            return None

    def extract_candidate_profile(self, cv_text: str) -> CandidateProfileInput | None:
        if not self._client:
            return None

        system_prompt = (
            "Eres un extractor de curriculum vitae para un formulario de registro. "
            "Debes devolver SOLO JSON valido, sin markdown, sin explicaciones y sin texto extra. "
            "El JSON debe contener estas claves: displayName, professionalTitle, summary, skills, "
            "experience, education, location, nationality, languages, sector, expectedSalary, availability, "
            "email, phoneNumber, github, linkedin, links. "
            "No devuelvas un objeto vacio. Si un campo no aparece, usa null o [] segun corresponda. "
            "Si el CV permite inferir un dato de forma razonable, rellena el valor. "
            "El campo sector debe ser EXACTAMENTE uno de estos valores: "
            "Tecnologia, Finanzas, Salud, Educacion, Retail, Logistica, Marketing, "
            "Construccion, Energia, Servicios, Telecomunicaciones, Otro. "
            "Si no se puede inferir un sector claro, usa 'Otro'. "
            "Para languages usa nombres completos (por ejemplo: Espanol, English), no codigos como es/en. "
            "Para links devuelve una lista de URLs profesionales validas (GitHub, LinkedIn, portafolio, etc.). "
            "Para experience.start, experience.end, education.start y education.end usa formato YYYY-MM-DD. "
            "Si solo existe el ano usa YYYY-01-01."
        )

        user_prompt = (
            "Extrae y normaliza la siguiente hoja de vida en JSON. "
            "Tu respuesta debe seguir este ejemplo de estructura:\n"
            "{\n"
            '  "displayName": "Juan Felipe Perez",\n'
            '  "professionalTitle": "Backend Engineer",\n'
            '  "summary": "Resumen corto del candidato",\n'
            '  "skills": ["Python", "FastAPI"],\n'
            '  "experience": [\n'
            '    {\n'
            '      "role": "Backend Developer",\n'
            '      "project": "Order Management System",\n'
            '      "company": "Freelance",\n'
            '      "tech": ["Java", "Spring Boot", "PostgreSQL"],\n'
            '      "description": "Desarrollo backend y APIs REST",\n'
            '      "start": "2023-01-01",\n'
            '      "end": "2024-01-01"\n'
            '    }\n'
            '  ],\n'
            '  "education": [\n'
            '    {\n'
            '      "institution": "Escuela Colombiana de Ingenieria Julio Garavito",\n'
            '      "degree": "Ingenieria de Sistemas",\n'
            '      "start": "2022-01-01",\n'
            '      "end": "2026-01-01",\n'
            '      "status": "En curso"\n'
            '    }\n'
            '  ],\n'
            '  "location": "Bogota",\n'
            '  "nationality": "Colombia",\n'
            '  "languages": ["Espanol", "English"],\n'
            '  "sector": "Tecnologia",\n'
            '  "expectedSalary": 7000000,\n'
            '  "availability": null,\n'
            '  "email": "persona@email.com",\n'
            '  "phoneNumber": "+57 300 000 0000",\n'
            '  "github": ["https://github.com/usuario"],\n'
            '  "linkedin": ["https://linkedin.com/in/usuario"],\n'
            '  "links": ["https://github.com/usuario", "https://linkedin.com/in/usuario", "https://miportafolio.com"]\n'
            "}\n\n"
            "Devuelve exclusivamente JSON valido, sin markdown ni explicaciones.\n\n"
            f"CV:\n{cv_text}"
        )

        try:
            if self._use_project_responses_api:
                response = self._client.responses.create(
                    model=self.chat_deployment,
                    instructions=system_prompt,
                    input=user_prompt,
                    max_output_tokens=1200,
                )
                content = getattr(response, "output_text", "{}") or "{}"
                data = self._safe_json_loads(content)
                data = self._normalize_extracted_profile(data)
                return CandidateProfileInput.model_validate(data)

            completion = self._client.chat.completions.create(
                model=self.chat_deployment,
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            if not completion.choices:
                return None

            content = completion.choices[0].message.content or "{}"
            data = self._safe_json_loads(content)
            data = self._normalize_extracted_profile(data)
            return CandidateProfileInput.model_validate(data)
        except Exception:
            return None

    def _normalize_extracted_profile(self, data: dict) -> dict:
        if not isinstance(data, dict):
            return {}

        sector_value = data.get("sector")
        if isinstance(sector_value, str):
            normalized = self._normalize_sector(sector_value)
            data["sector"] = normalized
        elif sector_value is not None:
            data["sector"] = "Otro"

        return data

    def _normalize_sector(self, raw_sector: str) -> str:
        cleaned = (raw_sector or "").strip()
        if not cleaned:
            return "Otro"

        lowered = cleaned.lower()
        for sector in self.ALLOWED_SECTORS:
            if lowered == sector.lower():
                return sector

        synonyms = {
            "it": "Tecnologia",
            "tech": "Tecnologia",
            "software": "Tecnologia",
            "saas": "Tecnologia",
            "banking": "Finanzas",
            "banca": "Finanzas",
            "medical": "Salud",
            "health": "Salud",
            "education": "Educacion",
            "logistics": "Logistica",
            "construction": "Construccion",
            "energy": "Energia",
            "telecom": "Telecomunicaciones",
        }
        if lowered in synonyms:
            return synonyms[lowered]

        return "Otro"



    def analyze_candidate_profile(self, candidate_profile: dict) -> dict:
            """
            Analyze a candidate profile combining static rules and LLM feedback.
            Returns a dict with overall_score, summary, sections and suggestions.
            """
            static_suggestions, overall_score, short_summary = self._build_static_profile_feedback(candidate_profile)
            ai_result = self._get_ai_profile_feedback(candidate_profile)
            return self._merge_profile_feedback(static_suggestions, ai_result, overall_score, short_summary)
    
    def _safe_json_loads(self, content: str) -> dict:
        raw = (content or "").strip()
        if not raw:
            return {}

        fence_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            raw,
            re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            raw = fence_match.group(1).strip()

        start = raw.find("{")
        end = raw.rfind("}")

        if start != -1 and end != -1 and end >= start:
            raw = raw[start:end + 1]

        try:
            return json.loads(raw)
        except Exception:
            return {}

