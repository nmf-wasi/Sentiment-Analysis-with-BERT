# Emotion Classification API (DistilBERT + FastAPI)

A small REST API that serves a Hugging Face DistilBERT emotion classifier. 

> **Note:** the model is a pretrained community model (`nmfairuz/distilbert-emotion-classifier`). The goal of this project was the serving, testing and deployment work, not model quality.

## Features

- `POST /predict` returns the predicted emotion label and a confidence score
- `GET /health` returns `200` only when the model is loaded, and `503` otherwise
- Model loads in a background thread at startup, so the server accepts requests immediately and reports its state honestly
- Input validation with Pydantic (empty or missing text is rejected with `422`)
- Automated tests with pytest
- Dockerized with CPU-only PyTorch to keep the image small
- Deployed once on Railway (the deployment has since been taken down)

## Example

Request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am frustrated!"}'
```

Response:

```json
{
  "label": "anger",
  "confidence": 0.9979089498519897
}
```

Health check:

```bash
curl http://localhost:8000/health
# 200 {"status": "ok"}   once the model is loaded
# 503                    while loading or if loading failed
```

## API

| Method | Path       | Body                  | Success             | Errors                                                            |
| ------ | ---------- | --------------------- | ------------------- | ----------------------------------------------------------------- |
| POST   | `/predict` | `{"text": "<string>"}` | `200` label + score | `422` invalid or empty text, `503` model not ready or failed      |
| GET    | `/health`  | none                  | `200` `{"status": "ok"}` | `503` model loading or failed (includes the error message)   |

Interactive docs are available at `/docs` when the server is running.

## Configuration

The app reads two environment variables (from a local `.env` file, or from the platform's variables when deployed):

| Variable   | Description                                             |
| ---------- | ------------------------------------------------------- |
| `MODEL_ID` | Hugging Face model repo id, e.g. `nmfairuz/distilbert-emotion-classifier` |
| `HF_TOKEN` | Hugging Face access token (read-only is enough)         |

Example `.env`:

```
MODEL_ID=nmfairuz/distilbert-emotion-classifier
HF_TOKEN=your_token_here
```

When setting these in a hosting dashboard, enter the values **without quotes**. A dashboard treats quotes as part of the value, and the model id will be rejected as invalid.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

The first start downloads the model, so `/health` returns `503` until loading finishes.

## Run with Docker

```bash
docker compose up --build
```

Or without compose:

```bash
docker build -t emotion-api .
docker run -p 8000:8000 --env-file .env emotion-api
```

The container listens on `$PORT` if the platform sets it, and falls back to `8000` otherwise.

## Tests

```bash
pip install pytest httpx
pytest -v
```

Tests cover:

- a valid request returns `200` with `label` and `confidence`
- an empty string returns `422`
- a missing `text` field returns `422`
- `/health` returns `200` with `{"status": "ok"}`

A session-scoped fixture starts the app (so the lifespan runs), waits until `/health` reports ready, and shares one loaded model across all tests. The tests use the real model, so they need network access on the first run.

## Design decisions

**Background model loading instead of crashing at startup.** Loading blocks for a while on a cold start (it downloads the model). I load it in a worker thread from the FastAPI `lifespan` handler, so the server comes up immediately. A shared `state` records `loading`, `ready` or `failed`, and both `/predict` and `/health` return `503` unless the state is `ready`. This keeps traffic away from a container with no model, and a failed load stays visible, with the error message, instead of looking like a healthy app.

**Lifespan instead of module-level loading.** Importing the module no longer triggers a model download, which keeps tests and tooling fast, and gives the model a defined startup and shutdown.

**A health check that reads state, not the model.** `/health` is called repeatedly by the platform, so it only reads the in-memory state and never runs inference.

**CPU-only PyTorch in Docker.** The service is for small-scale inference, and the CPU build avoids a multi-gigabyte image.

## Deployment notes (Railway)

I deployed this on Railway from the Dockerfile to practice the workflow. Things that mattered:

- Use the `$PORT` variable the platform assigns, rather than a hardcoded port
- Set `HF_TOKEN` and `MODEL_ID` in the platform's variables, without quotes
- Set the health check path to `/health` and give it a generous timeout, since the model takes time to download on first boot

## Project structure

```
src/main.py          FastAPI app, lifespan loading, /predict and /health
tests/               pytest suite and shared client fixture
Dockerfile           CPU-only PyTorch image
docker-compose.yml   Local container setup
requirements.txt     Runtime dependencies
pytest.ini           Test configuration
```

## Possible improvements

- Cache the model in the image or a volume to avoid re-downloading on each deploy
- Retry loading on failure instead of staying in the `failed` state until restart
- Fine-tune on a dataset of my own instead of using a pretrained community model
