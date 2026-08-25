import json
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from model import SimpleClassifier
except ImportError:
    from src.model import SimpleClassifier

app = FastAPI(title="ML Model Serving API")
model_path = Path(__file__).resolve().parent.parent / "artifacts" / "model.pt"
model = SimpleClassifier()


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_path: str


@app.on_event("startup")
def load_model():
    if model_path.exists():
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded model from {model_path}")
    else:
        print(f"No model found at {model_path}; using untrained model.")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    x = torch.tensor(payload.features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        prediction = int(torch.argmax(probs).item())
        probability = float(torch.max(probs).item())
    return PredictionResponse(prediction=prediction, probability=probability, model_path=str(model_path))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
