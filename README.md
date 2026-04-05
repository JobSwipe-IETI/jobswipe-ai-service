# JobSwipe AI Service

Servicio de IA para JobSwipe encargado de:

- calcular compatibilidad entre candidatos y vacantes
- generar embeddings semánticos
- extraer un perfil estructurado desde un CV en PDF
- producir feedback explicativo para apoyar el proceso de selección

El servicio está construido con FastAPI y Azure OpenAI, y está pensado para trabajar junto con el backend principal de JobSwipe, que envía perfiles estructurados de candidato y vacante.

## Objetivo

Este microservicio resuelve dos necesidades del producto:

1. Convertir información profesional en señales útiles de matching.
2. Preprocesar información de candidatos para facilitar el onboarding y el análisis de perfiles.

La lógica actual combina:

- embeddings para similitud semántica
- evaluación estructurada por IA para dimensiones como tecnología, experiencia, requisitos y contexto
- reglas de negocio controladas para penalizaciones críticas como tecnologías faltantes, seniority, salario y ubicación

Esto permite obtener un score más realista, explicable y estable que un porcentaje basado únicamente en similitud semántica.

## Stack

- Python 3.14
- FastAPI
- Uvicorn
- Azure OpenAI
- Pytest

## Estructura del proyecto

```text
app/
  controllers/
    embedding_controller.py
    profile_controller.py
  models/
    schemas.py
  services/
    azure_llm_service.py
    embedding_service.py
    matching_service.py
    pdf_extraction_service.py
  main.py

tests/
  test_api_controller.py
  test_azure_llm_service.py
  test_embedding_service.py
  test_matching_service.py
  test_pdf_extraction_service.py
  test_profile_controller.py
  test_schemas.py
```

## Configuración local

### 1. Variables de entorno

Crea un archivo `.env` en la raíz del proyecto con estas variables:

```env
AZURE_OPENAI_API_KEY=tu_api_key
AZURE_OPENAI_ENDPOINT=https://ieti-ia.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=IETI-IA
AZURE_OPENAI_API_VERSION=2023-05-15

AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_CHAT_API_VERSION=2024-10-21

# Opcional: recurso separado para chat/PDF
AZURE_OPENAI_PDF_API_KEY=tu_api_key_pdf
AZURE_OPENAI_PDF_ENDPOINT=https://tu-recurso-pdf.openai.azure.com/
AZURE_OPENAI_PDF_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_PDF_API_VERSION=2024-10-21
```

Notas:

- `AZURE_OPENAI_DEPLOYMENT` se usa para embeddings.
- `AZURE_OPENAI_CHAT_DEPLOYMENT` y/o `AZURE_OPENAI_PDF_DEPLOYMENT` se usan para evaluación con LLM y extracción desde PDF.
- Si no defines variables dedicadas para PDF/chat, el servicio hace fallback a las variables generales.

### 2. Instalar dependencias

```powershell
cd C:\dev\jobswipe-ai-service
C:\dev\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si tu entorno no tiene `python-dotenv`, instálalo también:

```powershell
C:\dev\.venv\Scripts\python.exe -m pip install python-dotenv
```

### 3. Ejecutar el servicio

Opción recomendada:

```powershell
cd C:\dev\jobswipe-ai-service
C:\dev\.venv\Scripts\Activate.ps1
.\start.ps1
```

O manualmente:

```powershell
cd C:\dev\jobswipe-ai-service
C:\dev\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn app.main:app --reload --port 8000
```

El servicio quedará disponible en:

```text
http://localhost:8000
```

## Endpoints principales

## `POST /embeddings/match`

Calcula la compatibilidad entre un candidato y una vacante.

Este es el endpoint principal del servicio.

### Cómo funciona

Cuando recibe perfiles estructurados:

1. Convierte el perfil del candidato y la vacante a texto de trabajo.
2. Genera embeddings para ambos textos.
3. Calcula similitud semántica.
4. Pide al LLM una evaluación estructurada por dimensiones:
   - `technology_fit`
   - `experience_fit`
   - `requirements_fit`
   - `context_fit`
5. Calcula un score final controlado por el servicio.
6. Aplica penalizaciones duras si faltan requisitos críticos.
7. Devuelve porcentaje, nivel, feedback y breakdown detallado.

### Formas soportadas de request

#### Opción A: perfiles estructurados

Recomendada para integración con `Backend_JobSwipe`.

```json
{
  "candidateProfile": {
    "professionalTitle": "Backend Engineer",
    "summary": "Ingeniero backend con experiencia en Python.",
    "skills": ["Python", "FastAPI", "PostgreSQL"],
    "experience": [
      {
        "role": "Backend Developer",
        "project": "Payments API",
        "company": "Acme",
        "tech": ["Python", "FastAPI", "PostgreSQL"],
        "description": "Desarrollo de APIs REST y servicios backend.",
        "start": "2021-01-01",
        "end": "2024-01-01"
      }
    ],
    "education": [
      {
        "institution": "Universidad Nacional",
        "degree": "Ingenieria de Sistemas",
        "start": "2016-01-01",
        "end": "2021-01-01",
        "status": "Graduated"
      }
    ],
    "location": "Bogota",
    "languages": ["English", "Spanish"],
    "expectedSalary": 8000000,
    "availability": "Immediate",
    "email": "persona@email.com",
    "phoneNumber": "+57 300 000 0000",
    "github": "https://github.com/usuario",
    "linkedin": "https://linkedin.com/in/usuario"
  },
  "vacancyProfile": {
    "title": "Senior Python Backend Engineer",
    "description": "Rol para construir APIs escalables en Python y FastAPI.",
    "location": "Bogota",
    "modality": "HYBRID",
    "employmentType": "FULL_TIME",
    "experienceLevel": "SENIOR",
    "technologies": ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "softSkills": ["Communication", "Ownership"],
    "responsibilities": ["Build APIs", "Code reviews", "System design"],
    "technicalRequirements": ["English", "REST APIs", "Microservices"],
    "minSalary": 7000000,
    "maxSalary": 9500000
  }
}
```

#### Opción B: texto libre

Compatible con pruebas rápidas o integraciones simples.

```json
{
  "candidate_text": "Backend developer with Python and FastAPI experience",
  "vacancy_text": "Senior Python backend engineer focused on APIs and microservices"
}
```

#### Opción C: formato legacy

Se mantiene por compatibilidad:

```json
{
  "text1": "candidate text",
  "text2": "vacancy text"
}
```

### Respuesta

```json
{
  "similarity_score": 0.9174,
  "compatibility_percentage": 67.0,
  "compatibility_level": "medium",
  "feedback": "Resumen explicativo del match",
  "used_llm_feedback": true,
  "score_breakdown": {
    "semantic_score": 100.0,
    "technology_score": 70.0,
    "experience_score": 85.0,
    "requirements_score": 80.0,
    "context_score": 75.0,
    "compatibility_percentage": 67.0,
    "hard_requirements": {
      "passed": false,
      "penalty_points": 10.0,
      "penalties": [
        {
          "type": "missing_technologies",
          "penalty": 10.0,
          "details": ["fastapi"]
        }
      ],
      "reasons": ["Faltan tecnologias clave: fastapi"]
    },
    "details": {
      "technology": {
        "matched": ["Python", "PostgreSQL", "Docker"],
        "missing": ["FastAPI"],
        "rationale": "Explicacion de la IA"
      },
      "experience": {
        "matched": ["Backend development", "System architecture"],
        "missing": [],
        "rationale": "Explicacion de la IA"
      },
      "requirements": {
        "matched": ["English", "REST APIs", "Microservices"],
        "missing": ["FastAPI"],
        "rationale": "Explicacion de la IA",
        "source": "llm",
        "hard": {
          "matched": ["English", "REST APIs", "Microservices"],
          "missing": ["FastAPI"]
        },
        "soft": {
          "matched": [],
          "missing": []
        },
        "fallback_assessment": {
          "matched": ["python", "docker"],
          "missing": ["fastapi"]
        }
      },
      "context": {
        "matched": ["Location: Bogota", "Modality: HYBRID"],
        "missing": [],
        "rationale": "Explicacion de la IA"
      }
    },
    "llm_evaluation_used": true,
    "llm_red_flags": [
      {
        "type": "missing_technology",
        "severity": "high",
        "detail": "Falta FastAPI"
      }
    ],
    "llm_red_flag_penalty": 0.0
  }
}
```

### Interpretación del resultado

- `similarity_score`: similitud coseno entre embeddings
- `compatibility_percentage`: score final calculado por el servicio
- `compatibility_level`:
  - `high`: 75 o más
  - `medium`: 50 a 74.99
  - `low`: menor a 50
- `feedback`: explicación resumida para producto o debugging
- `score_breakdown`: detalle interno del cálculo

### Qué influye en el score

- tecnologías
- experiencia
- requisitos de la vacante
- contexto del rol
- penalizaciones críticas

Penalizaciones típicas:

- tecnologías clave faltantes
- gap de seniority
- salario fuera de rango
- incompatibilidad fuerte de ubicación/modalidad

## `POST /profiles/extract-cv`

Extrae un perfil estructurado a partir de un CV en PDF.

Este endpoint está pensado para prellenar formularios de onboarding de candidatos.

### Request

`multipart/form-data`

Campo esperado:

- `file`: archivo PDF

Ejemplo con PowerShell:

```powershell
curl.exe -X POST "http://localhost:8000/profiles/extract-cv" `
  -F "file=@C:\ruta\cv.pdf"
```

### Response

```json
{
  "candidateProfile": {
    "displayName": "Juan Perez",
    "professionalTitle": "Backend Engineer",
    "summary": "Resumen profesional",
    "skills": ["Python", "FastAPI"],
    "experience": [],
    "education": [],
    "location": "Bogota",
    "languages": ["Spanish", "English"],
    "expectedSalary": 7000000,
    "availability": null,
    "email": "juan@email.com",
    "phoneNumber": "+57 300 000 0000",
    "github": ["https://github.com/usuario"],
    "linkedin": ["https://linkedin.com/in/usuario"],
    "links": ["https://github.com/usuario", "https://linkedin.com/in/usuario"]
  },
  "rawTextLength": 2840,
  "usedAi": true
}
```

### Notas importantes

- funciona mejor con PDFs que contienen texto seleccionable
- si el PDF es escaneado, puede requerir OCR antes
- el resultado está pensado para ser revisado por el usuario antes de guardarlo

## Endpoint auxiliar

## `POST /embeddings/`

Genera el embedding de un texto.

### Request

```json
{
  "text": "Python developer with backend experience"
}
```

### Response

```json
{
  "result": [0.001, -0.002, 0.003]
}
```

## Ejemplos de prueba

## Probar matching

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/embeddings/match" `
  -Method Post `
  -ContentType "application/json" `
  -Body (Get-Content .\sample-match.json -Raw)
```

## Probar extracción de CV

```powershell
curl.exe -X POST "http://localhost:8000/profiles/extract-cv" `
  -F "file=@C:\ruta\cv.pdf"
```

## Pruebas automáticas

Ejecuta la suite relevante con:

```powershell
cd C:\dev\jobswipe-ai-service
C:\dev\.venv\Scripts\python.exe -m pytest
```

Durante esta mejora del servicio se validaron especialmente:

```powershell
C:\dev\.venv\Scripts\python.exe -m pytest tests/test_matching_service.py tests/test_api_controller.py tests/test_azure_llm_service.py
```

## Integración con JobSwipe

Este servicio está diseñado para integrarse con:

- `Frontend_JobSwipe`, que puede consumir `extract-cv`
- `Backend_JobSwipe`, que modela `candidateProfile` y `vacancyProfile`

La forma recomendada de integrarlo es usando perfiles estructurados, no texto libre.

## Estado actual

El servicio ya soporta:

- matching semántico con embeddings
- evaluación estructurada por dimensiones con LLM
- penalizaciones críticas controladas por el servicio
- extracción de perfil desde CV
- feedback explicativo
- fallback cuando el LLM no está disponible

## Despliegue en Azure

Puedes desplegarlo en Azure App Service configurando las variables de entorno y usando como startup command:

```text
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Si usas contenedor:

```dockerfile
FROM python:3.14
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
