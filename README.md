# Optimization-Driven Music Recommendation System

Matrix Factorization · Sparse Data · Convergence Analysis · MLOps Pipeline · Kubernetes · GitOps

See `notebooks/03_optimization_analysis.ipynb` for exploratory analysis and `src/` for runnable code.

## Quickstart (local)

```bash
git clone <your-repo-url>
cd music-recommender-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train/train.py --epochs 10
uvicorn src.api.app:app --reload --port 8000
```

## What to include in your CV / LOM

- Built a matrix-factorization recommender on sparse user–item data.
- Conducted optimization experiments: learning rate, batch size, regularization.
- Deployed model as a FastAPI service (Docker + Kubernetes) and added Prometheus/Grafana monitoring.


[ci] trigger: 2025-11-21 03:02:35Z

