# mlops-pytorch-pipeline

A minimal PyTorch MLOps project scaffold for training, serving, Dockerization, Kubernetes manifests, CI, and tests.

## Project structure

- `src/` — training, model, dataset, and serving code
- `configs/` — training configuration
- `docker/` — container definitions for training and serving
- `k8s/` — Kubernetes deployment artifacts
- `requirements/` — environment dependencies
- `tests/` — unit tests
- `.github/workflows/` — CI pipeline

## Quick start

1. Create a virtual environment.
2. Install dependencies:
   ```bash
   python3 -m pip install -r requirements/train.txt
   python3 -m pip install -r requirements/serve.txt
   python3 -m pip install pytest
   ```
3. Train the model:
   ```bash
   python3 src/train.py
   ```
4. Run the API:
   ```bash
   python3 src/serve.py
   ```
5. Run tests:
   ```bash
   pytest -q
   ```