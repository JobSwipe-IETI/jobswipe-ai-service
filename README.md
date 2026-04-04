# jobswipe-ai-service

Servicio de IA para JobSwipe. Genera embeddings y feedback de compatibilidad candidato-vacante usando Azure.

## Configuración local

### 1. Crear archivo `.env` en la raíz del proyecto

```
AZURE_OPENAI_API_KEY=tu_api_key_aqui
AZURE_OPENAI_ENDPOINT=https://ieti-ia.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=IETI-IA
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_CHAT_API_VERSION=2024-10-21

# Opcional: separar completamente el modelo de PDF/chat en otro recurso o key
AZURE_OPENAI_PDF_API_KEY=tu_api_key_pdf
AZURE_OPENAI_PDF_ENDPOINT=https://tu-recurso-pdf.openai.azure.com/
AZURE_OPENAI_PDF_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_PDF_API_VERSION=2024-10-21
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Levantar el servicio

**Opción A: Con script PowerShell (recomendado)**
```powershell
cd c:\dev\jobswipe-ai-service
.\start.ps1
```

**Opción B: Manual con variables seteadas**
```powershell
$env:AZURE_OPENAI_API_KEY = "tu_api_key"
$env:AZURE_OPENAI_ENDPOINT = "https://ieti-ia.cognitiveservices.azure.com/"
$env:AZURE_OPENAI_DEPLOYMENT = "IETI-IA"
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn app.main:app --reload --port 8000
```

## Despliegue en Azure

### Configurar en Azure App Service

1. En Azure Portal → Tu App Service → Settings → Environment variables
2. Añade las variables:
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_DEPLOYMENT`
   - `AZURE_OPENAI_API_VERSION`
  - `AZURE_OPENAI_CHAT_DEPLOYMENT`
  - `AZURE_OPENAI_CHAT_API_VERSION`
  - `AZURE_OPENAI_PDF_API_KEY` (opcional)
  - `AZURE_OPENAI_PDF_ENDPOINT` (opcional)
  - `AZURE_OPENAI_PDF_DEPLOYMENT` (opcional)
  - `AZURE_OPENAI_PDF_API_VERSION` (opcional)

3. Verifica que `.env` NO está en el repositorio (`.gitignore` debe excluirlo)

### Startup Command en Azure

En Azure App Service, configura:
```
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

O con Docker:
```dockerfile
FROM python:3.14
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Endpoints

### `POST /embeddings/`

Genera el embedding de un texto.

**Request:**
```json
{
  "text": "Python developer with 3 years of backend experience"
}
```

**Response:**
```json
{
  "result": [0.001, -0.002, ...]  // Vector de embeddings
}
```

---

### `POST /embeddings/match`

Compara candidato vs vacante. Devuelve porcentaje de compatibilidad y feedback.

**Request:**
```json
{
  "candidate_text": "Backend developer, Python, FastAPI, MongoDB, PostgreSQL, 5 años",
  "vacancy_text": "Senior Python backend engineer, FastAPI, databases, API design"
}
```

**Request estructurado (recomendado para Backend_JobSwipe):**
```json
{
  "candidate_profile": {
    "professional_title": "Backend Engineer",
    "summary": "Ingeniero backend con 5 años en Python y APIs.",
    "skills": ["Python", "FastAPI", "PostgreSQL", "MongoDB"],
    "experience": "Desarrollo de APIs REST y microservicios.",
    "education": "Ingenieria de Sistemas",
    "location": "Bogota",
    "languages": ["es", "en"],
    "expected_salary": 7000000,
    "availability": "Immediate"
  },
  "vacancy_profile": {
    "title": "Senior Python Backend Engineer",
    "description": "Rol para construir APIs y servicios de matching.",
    "location": "Bogota",
    "modality": "HYBRID",
    "employment_type": "FULL_TIME",
    "experience_level": "SENIOR",
    "technologies": ["Python", "FastAPI", "PostgreSQL"],
    "soft_skills": ["Communication", "Ownership"],
    "responsibilities": ["Build APIs", "Code reviews"],
    "technical_requirements": ["5+ years backend", "REST APIs"],
    "min_salary": 6000000,
    "max_salary": 9000000
  }
}
```

**Response:**
```json
{
  "similarity_score": 0.8231,
  "compatibility_percentage": 91.16,
  "compatibility_level": "high",
  "feedback": "El perfil del candidato está alineado con la vacante en stack de tecnología y experiencia. Fortalezas: Python, FastAPI, databases. Brecha: no menciona experiencia con sistemas distribuidos.",
  "used_llm_feedback": true
}
```

**Campos de respuesta:**
- `similarity_score`: Similitud coseno bruta [-1, 1]
- `compatibility_percentage`: Porcentaje normalizado [0, 100]
- `compatibility_level`: `high` (≥80%), `medium` (≥60%), `low` (<60%)
- `feedback`: Texto explicativo (LLM si está disponible, sino reglas)
- `used_llm_feedback`: Si fue generado por LLM (true) o reglas (false)

---

## Compatibilidad hacia atrás

Request legacy (sigue soportado):
```json
{
  "text1": "candidate text",
  "text2": "vacancy text"
}
```

### `POST /profiles/extract-cv`

Recibe un PDF del CV y devuelve un JSON con el perfil del candidato ya estructurado para prellenar el registro.

Request:
```multipart/form-data
file: <cv.pdf>
```

Response:
```json
{
  "candidateProfile": {
    "professionalTitle": "Backend Engineer",
    "summary": "...",
    "skills": ["Python", "FastAPI"],
    "experience": "...",
    "education": "...",
    "location": "Bogota",
    "languages": ["es", "en"],
    "expectedSalary": 7000000,
    "availability": "Immediate"
  },
  "rawTextLength": 2840,
  "usedAi": true
}
```

Nota: este endpoint funciona mejor con PDFs que tienen texto seleccionable. Si el PDF es escaneado, necesitas OCR antes de extraerlo.