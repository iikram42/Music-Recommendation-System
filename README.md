Music Recommendation System

Summary: 
  A reproducible end-to-end Matrix Factorization (MF) based recommendation system
  featuring a full training pipeline, experiment tooling, FastAPI inference API,
  containerization via Docker, Kubernetes deployment manifests, and CI/CD automation.
  This repository combines numerical experimentation with practical engineering design.

Goals:
  - "Implement a clean, minimal Matrix Factorization model from scratch"
  - "Analyze optimization behavior through experiments"
  - "Serve predictions through a FastAPI microservice"
  - "Package and deploy using Docker + Kubernetes"
  - "Automate testing and builds using GitHub Actions"

Features:
  matrix_factorization:
    file: "src/model/matrix_factorization.py"
    description: >
      SGD-based MF implementation with L2 regularization, configurable latent
      dimension k, learning rate η, and λ regularization.
  training_pipeline:
    file: "src/train/train.py"
    description: >
      Loads processed data, trains the MF model, logs loss across epochs,
      and exports trained weights into artifacts/.
  api:
    file: "src/api/app.py"
    endpoints:
      - "/health — readiness/liveness endpoint"
      - "/recommend/{user_id} — return top-N predicted items"
  experiments:
    folder: "experiments/"
    scripts:
      - "gradient_norms.py — gradient norm evolution across updates"
      - "lr_analysis.py — effect of η on stability/convergence"
      - "reg_analysis.py — impact of λ on under/overfitting"
      - "latent_drift.py — latent vector drift behavior"
    notebooks:
      - "notebooks/03_optimization_analysis.ipynb"
      - "notebooks/optimization_experiments.ipynb"
  tests:
    folder: "tests/"
    description: "Pytest-based tests verifying MF shapes, decreasing loss, and API behavior."

recommended_artifacts:
  - "artifacts/loss.png"
  - "artifacts/gradient_norm.png"
  - "artifacts/latent_drift.png"
  - "artifacts/sample_data.csv (processed preview only)"
  - "artifacts/model.joblib (optional, excluded via .gitignore)"

quick_start:
  prerequisites:
    - "Python 3.11+"
    - "Docker"
    - "kubectl (optional)"
  steps:
    - "git clone https://github.com/iikram42/Music-Recommendation-System"
    - "cd Music-Recommendation-System"
    - "python -m venv .venv"
    - "source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1"
    - "pip install -r requirements.txt"
    - "python src/train/train.py --epochs 5 --batch 32 --lr 0.01 --reg 0.01"
    - "uvicorn src.api.app:app --reload --port 8000"
  api_test:
    - "curl http://127.0.0.1:8000/health"
    - "curl http://127.0.0.1:8000/recommend/1"

Docker:
  build: "docker build -t iikram42/music-recommendation-system:latest ."
  run: "docker run --rm -p 8000:8000 iikram42/music-recommendation-system:latest"

Kubernetes:
  manifests:
    - "k8s/deployment.yaml"
    - "k8s/service.yaml"
  deploy:
    - "kubectl apply -f k8s/deployment.yaml"
    - "kubectl apply -f k8s/service.yaml"
  port_forward:
    - "kubectl port-forward svc/recommender-service 8080:80"
  test:
    - "curl http://127.0.0.1:8080/health"
    - "curl http://127.0.0.1:8080/recommend/1"

architecture_diagram: |
  +-------------------------+
  |   FastAPI Microservice  |
  |   /health  /recommend   |
  +------------+------------+
               |
               v
  +-------------------------+
  |  Matrix Factorization   |
  |     (P, Q Embeddings)   |
  +------------+------------+
               |
               v
  +-------------------------+
  |       Docker Image      |
  +------------+------------+
               |
               v
  +---------------------------------------------+
  |           Kubernetes Deployment             |
  |  - Readiness/Liveness Probes                |
  |  - Replica management                       |
  +-------------------+-------------------------+
                       |
                       v
  +-------------------------+
  |   ClusterIP Service     |
  +-------------------------+

Project_structure: |
  .
  ├── src/
  │   ├── api/app.py
  │   ├── model/matrix_factorization.py
  │   └── train/train.py
  ├── experiments/
  ├── artifacts/
  ├── data/
  ├── k8s/
  │   ├── deployment.yaml
  │   └── service.yaml
  ├── notebooks/
  ├── tests/
  ├── requirements.txt
  ├── Dockerfile
  ├── .github/workflows/ci-cd.yml
  └── README.md

CI/CD:
  file: ".github/workflows/ci-cd.yml"
  tasks:
    - "Run pytest"
    - "Build Docker image"
    - "Push to Docker Hub using DOCKERHUB_USERNAME / DOCKERHUB_TOKEN"
    - "Optional: Deploy to Kubernetes using KUBE_CONFIG"

notes:
  - "Keep large artifacts (model.joblib, PNGs) out of Git — commit small examples only."
  - "Add plot captions in README when sharing results."
  - "If deploying publicly, add resource limits, scaling rules, and logging."
  - "FastAPI endpoints support async, so heavy ops should move to background tasks."

author: "iikram42"
license: "MIT"
repository: "https://github.com/iikram42/Music-Recommendation-System"
