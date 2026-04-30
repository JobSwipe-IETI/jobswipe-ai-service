FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Crear usuario (seguridad)
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser

# 🔥 1. copiar requirements primero
COPY requirements.txt ./

# 🔥 2. instalar deps (incluyendo torch CPU)
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# 🔥 3. copiar código
COPY app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3)"

USER appuser

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]