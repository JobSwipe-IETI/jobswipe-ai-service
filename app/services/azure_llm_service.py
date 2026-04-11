import json
import os
import re

from openai import AzureOpenAI, OpenAI

from app.models.schemas import CandidateProfileInput


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

    def _safe_json_loads(self, content: str) -> dict:
        raw = (content or "").strip()
        if not raw:
            return {}

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
        if fence_match:
            raw = fence_match.group(1).strip()

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end >= start:
            raw = raw[start : end + 1]

        try:
            return json.loads(raw)
        except Exception:
            return {}
