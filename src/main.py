import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from transformers import pipeline
from contextlib import asynccontextmanager

ml_models = {}
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
state = {"status": "loading", "error": None}


def load_model():
    try:
        ml_models["classifier"] = pipeline(
            "text-classification", model=MODEL_ID, token=HF_TOKEN
        )
        state["status"] = "ready"
    except Exception as e:
        state["status"] = "failed"
        state["error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup : runs once, before the serrver accepts requests
    task = asyncio.create_task(asyncio.to_thread(load_model))
    # runs the blocking load in a worker thread, so the event loop stays free and the server starts accepting requests immediately
    yield
    ml_models.clear()  # on shutown clears the models


app = FastAPI(lifespan=lifespan)


class TextIn(BaseModel):
    text: str = Field(
        min_length=1,
    )


@app.post("/predict")
def predict(payload: TextIn):
    if state["status"] != "ready":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model {state['status']}"
            + (f": {state['error']}" if state["error"] else ""),
        )
    result = ml_models["classifier"](payload.text)[0]
    return {
        "label": result["label"],
        "confidence": result["score"],
    }


@app.get("/health")
def health():
    if state["status"] == "ready":
        return {"status": "ok"}
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"status": state["status"], "error": state["error"]},
    )
