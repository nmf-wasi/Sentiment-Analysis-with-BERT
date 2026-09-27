import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
print("Token loaded:", HF_TOKEN[:8] if HF_TOKEN else "NONE")
classifier = pipeline(
    "text-classification",
    model="nmfairuz/distilbert-emotion-classifier",
    token=HF_TOKEN,
)


class TextIn(BaseModel):
    text: str


@app.post("/predict")
def predict(payload: TextIn):
    result = classifier(payload.text)[0]
    return {
        "label": result["label"],
        "confidence": result["score"],
    }
