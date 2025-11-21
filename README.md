# Music Recommendation System — Research + Engineering Edition

This project implements a **Matrix Factorization (MF) based recommendation engine** served through a **FastAPI microservice**, packaged in **Docker**, tested via **PyTest**, and deployed to **Kubernetes** with full **CI/CD automation**.

It is designed as a hybrid **research + production engineering** project, combining theoretical insights with cloud-ready implementation.

## 📘 1. Research Motivation

Real-world recommendation datasets share common properties:

- **Sparse** — most user–item interactions are missing  
- **Noisy** — implicit feedback, click behaviour, human inconsistency  
- **High-dimensional** — thousands of users & items  
- **Non-convex** — MF loss surfaces contain saddle points, flat regions, and irregular curvature  

This project is structured to study:

- SGD behaviour on sparse matrices  
- Learning rate influence on convergence  
- Latent factor drift under weak regularization  
- Gradient noise due to sparsity  
- Bias–variance behaviour in MF  
- Reconstruction error patterns  

## 📐 2. Matrix Factorization Overview

### Objective
We approximate a sparse rating matrix **R** with low-rank matrices:

    R ≈ P Qᵀ

Where:
- **P** = user latent factors  
- **Q** = item latent factors  

### Loss Function

    min_{P,Q} ∑_{(u,i)∈Ω} (R_ui - P_u Q_iᵀ)^2 + λ(||P||² + ||Q||²)

### SGD Update Rules

For an observed interaction (u,i):

    e_ui = R_ui - P_u Q_iᵀ

Updates:

    P[u] ← P[u] + η (e * Q[i] - λ * P[u])
    Q[i] ← Q[i] + η (e * P[u] - λ * Q[i])

**Behavior Observations**
- Too high LR → oscillations  
- Too low LR → slow convergence  
- Low regularization → latent drift  
- Excessive regularization → underfitting  

## 📊 3. Example Dataset Snapshot

```
User/Item →   I1   I2   I3   I4   I5
U1            5    .    .    4    .
U2            .    .    3    .    2
U3            .    1    .    .    .
```

`.` indicates **missing interaction**, typical in recommendation datasets.

## 🧩 4. System Architecture

```
+------------------------------+
|        FastAPI Service       |
|    /health   /recommend      |
+--------------+---------------+
               |
               v
+------------------------------+
|    Matrix Factorization      |
|      (P,Q Embeddings)        |
+--------------+---------------+
               |
               v
+------------------------------+
|         Docker Image         |
+--------------+---------------+
               |
               v
+---------------------------------------------+
|            Kubernetes Deployment             |
|     - Readiness / Liveness Probes            |
|     - Replica Management                     |
+----------------------+----------------------+
                       |
                       v
+------------------------------+
|        ClusterIP Service     |
+------------------------------+
```

## 🗂 5. Project Structure

```
project-root/
│
├── src/
│   ├── api/
│   │   └── app.py                      # FastAPI inference service
│   ├── model/
│   │   └── matrix_factorization.py     # MF implementation
│   └── train/
│       └── train.py                    # Training script
│
├── experiments/                        # Experiment scripts & analysis
├── artifacts/                          # Loss curves, gradient norms, drift plots
├── data/                               # Sample datasets
│
├── k8s/
│   ├── deployment.yaml                 # Kubernetes Deployment
│   └── service.yaml                    # Kubernetes Service
│
├── notebooks/                          # Jupyter notebooks
├── tests/                              # Unit tests
│
├── requirements.txt
├── Dockerfile
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml                   # CI/CD pipeline
│
└── README.md
```

## 🧪 6. Local Development

### Build the image

```bash
docker build -t music-reco:latest .
```

### Run container

```bash
docker run -p 8000:8000 music-reco:latest
```

### Test API

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/recommend/1
```

## ☸️ 7. Kubernetes Deployment

### Apply manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Port-forward:

```bash
kubectl port-forward svc/recommender-service 8000:80
```

### Test:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/recommend/1
```

## 🔁 8. CI/CD Pipeline (GitHub Actions)

Pipeline includes:

- Install dependencies  
- Run unit tests  
- Build Docker image  
- Push to Docker Hub  
- Deploy to Kubernetes (optional if KUBE_CONFIG is set)  

## 📜 9. License

**MIT License** — free to use, modify, and distribute.
