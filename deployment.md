# Deployment

The only deployable component is the stateless HTTP API (`agentq.api.app:app`). It needs no database, cache or message queue, so a single container is enough and no docker-compose file is provided.

## Docker

```bash
docker build -t agentq:latest .
docker run --rm -p 8000:8000 agentq:latest
curl -s localhost:8000/health
```

- **Base image:** `python:3.12-slim`, running as a non-root user with a health check.
- **Profiles:** loaded from `/app/configs/profiles.yaml` (override with `AGENTQ_PROFILES` and a volume mount).
- **Request limit:** `AGENTQ_MAX_GRID_POINTS` bounds the per-request grid size (default 60,000).

## Without Docker

```bash
pip install -e ".[api]"
agentq serve --host 0.0.0.0 --port 8000
# or: uvicorn agentq.api.app:app --host 0.0.0.0 --port 8000 --workers 2
```

## Operational notes

- **CPU and latency:** each `/predict` call is CPU-bound, roughly 0.3–1 s on the paper's grid; `/capacity` runs about 40 predictions. Run several uvicorn workers, or put long sweeps behind a job queue if needed.
- **No authentication:** the API is intended for internal use. Put it behind an authenticating reverse proxy if exposed.
- **Disclaimer:** the API reports predictions with validity flags. It does not make provisioning decisions. See `SECURITY.md` and the limitations in `README.md`.
