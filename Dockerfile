# HTTP API image for agentq (CPU only, no GPU needed).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENTQ_PROFILES=/app/configs/profiles.yaml

WORKDIR /app
COPY pyproject.toml README.md requirements.txt requirements-api.txt ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt \
    && pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 agentq
USER agentq

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "agentq.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
