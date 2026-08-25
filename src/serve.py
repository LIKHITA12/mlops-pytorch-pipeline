import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

try:
    # prefer local import when run as package
    from model import get_model
except Exception:
    from src.model import get_model


app = FastAPI(title="ML Model Serving API")
base_artifacts = Path(__file__).resolve().parent.parent / "artifacts"
# prefer top-level model path, fall back to checkpoint dir used by training
model_path = base_artifacts / "model.pt"
checkpoint_path = base_artifacts / "checkpoints" / "model.pt"
if not model_path.exists() and checkpoint_path.exists():
    model_path = checkpoint_path

model = None
label_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_preprocess(image_size: int = 32):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


@app.on_event("startup")
def load_model():
    global model
    # Default: try to create a ResNet-18 for 10 classes
    try:
        model = get_model(architecture="resnet18", num_classes=10)
    except Exception:
        # fallback to simple MLP for environments without torchvision
        model = get_model(architecture="mlp", input_dim=10, hidden_dim=32, num_classes=10)

    if model_path.exists():
        ckpt = torch.load(model_path, map_location="cpu")
        # ckpt may be a state_dict or a dict containing 'model_state_dict'
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and any(k.startswith("module.") or k in model.state_dict() for k in ckpt.keys()):
            # assume it's a state_dict
            state = ckpt
        else:
            state = None

        if state is not None:
            try:
                model.load_state_dict(state)
                model.eval()
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load state_dict from {model_path}: {e}")
        else:
            print(f"Checkpoint at {model_path} did not contain a recognizable state dict.")
    else:
        print(f"No model found at {model_path}; using untrained model.")


@app.get("/health")
def health():
    return JSONResponse({"ok": model is not None})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    preprocess = get_preprocess(image_size=32)
    x = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()

    classes = [{"label": label_names[i], "probability": float(probs[i])} for i in range(len(probs))]
    top_idx = int(torch.argmax(torch.tensor(probs)).item())
    return JSONResponse({
        "prediction": label_names[top_idx],
        "probabilities": classes,
        "model_path": str(model_path),
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
