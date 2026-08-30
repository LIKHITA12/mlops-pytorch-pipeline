import io
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Ensure project root is on sys.path so imports work seamlessly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

try:
    from dataset import CIFAR10_CLASSES, get_transforms
    from model import get_model
except ImportError:
    from src.dataset import CIFAR10_CLASSES, get_transforms
    from src.model import get_model

# Global state
model: Optional[torch.nn.Module] = None
is_model_loaded: bool = False
loaded_model_path: Optional[str] = None
model_architecture: str = "resnet18"
num_classes: int = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to load model on startup."""
    global model, is_model_loaded
    success = load_model_checkpoint()
    if not success:
        print("Warning: No valid model checkpoint found at startup. /health will return 503 until a checkpoint is loaded.")
    yield


app = FastAPI(title="MLOps PyTorch Model Serving API", lifespan=lifespan)



def find_checkpoint_path() -> Optional[Path]:
    """Search potential locations for model checkpoint."""
    # 1. Direct environment variable
    if "MODEL_PATH" in os.environ:
        p = Path(os.environ["MODEL_PATH"])
        if p.exists() and p.is_file():
            return p

    # 2. Candidate paths
    candidate_paths = [
        Path("/app/checkpoints/classifier_v1.pt"),
        Path("/app/checkpoints/model.pt"),
        _PROJECT_ROOT / "checkpoints" / "classifier_v1.pt",
        _PROJECT_ROOT / "checkpoints" / "model.pt",
        _PROJECT_ROOT / "artifacts" / "checkpoints" / "classifier_v1.pt",
        _PROJECT_ROOT / "artifacts" / "checkpoints" / "model.pt",
        _PROJECT_ROOT / "artifacts" / "model.pt",
    ]

    for p in candidate_paths:
        if p.exists() and p.is_file():
            return p

    # 3. Check checkpoint dirs for any .pt or .pth file
    search_dirs = [
        Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")),
        _PROJECT_ROOT / "checkpoints",
        _PROJECT_ROOT / "artifacts" / "checkpoints",
    ]
    for d in search_dirs:
        if d.exists() and d.is_dir():
            pt_files = sorted(list(d.glob("*.pt")) + list(d.glob("*.pth")))
            if pt_files:
                return pt_files[0]

    return None


def load_model_checkpoint(checkpoint_path: Optional[str | Path] = None) -> bool:
    """Load model weights from checkpoint into global model."""
    global model, is_model_loaded, loaded_model_path, model_architecture, num_classes

    target_path = Path(checkpoint_path) if checkpoint_path else find_checkpoint_path()
    if target_path is None or not target_path.exists():
        is_model_loaded = False
        return False

    try:
        checkpoint = torch.load(target_path, map_location="cpu")
        arch = "resnet18"
        n_classes = 10

        # Extract metadata if available
        if isinstance(checkpoint, dict):
            arch = checkpoint.get("architecture", arch)
            n_classes = int(checkpoint.get("num_classes", n_classes))
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model_architecture = arch
        num_classes = n_classes
        model_instance = get_model(architecture=arch, num_classes=n_classes)
        model_instance.load_state_dict(state_dict)
        model_instance.eval()

        model = model_instance
        is_model_loaded = True
        loaded_model_path = str(target_path)
        print(f"Successfully loaded model from {loaded_model_path}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint from {target_path}: {e}")
        is_model_loaded = False
        return False


@app.get("/health")
def health():
    """Health check endpoint: returns 200 if model is loaded, 503 otherwise."""
    global is_model_loaded
    if not is_model_loaded:
        # Retry loading in case a checkpoint was recently saved/mounted
        load_model_checkpoint()

    if is_model_loaded and model is not None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "model_loaded": True,
                "model_path": loaded_model_path,
                "architecture": model_architecture,
                "num_classes": num_classes,
            },
        )

    return JSONResponse(
        status_code=503,
        content={
            "status": "unhealthy",
            "model_loaded": False,
            "detail": "Model checkpoint is not loaded.",
        },
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Accept an image file, run inference, and return class probabilities."""
    global model, is_model_loaded
    if not is_model_loaded or model is None:
        # Retry loading checkpoint
        if not load_model_checkpoint():
            raise HTTPException(status_code=503, detail="Model checkpoint is not loaded.")

    # Read image contents
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    tensor = preprocess(pil_image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].tolist()

    top_idx = int(torch.argmax(torch.tensor(probs)).item())
    labels = CIFAR10_CLASSES if num_classes == len(CIFAR10_CLASSES) else [f"class_{i}" for i in range(num_classes)]
    predicted_label = labels[top_idx] if top_idx < len(labels) else str(top_idx)

    probabilities_list = [
        {"class_id": i, "label": labels[i] if i < len(labels) else str(i), "probability": round(float(probs[i]), 6)}
        for i in range(len(probs))
    ]

    return JSONResponse({
        "status": "success",
        "predicted_class_id": top_idx,
        "prediction": predicted_label,
        "confidence": round(float(probs[top_idx]), 6),
        "probabilities": probabilities_list,
        "model_path": loaded_model_path,
    })


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

