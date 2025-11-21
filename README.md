# Music Recommendation System 🎵
A lightweight, production-ready recommendation microservice built with FastAPI, packaged with Docker, and deployable on Kubernetes.

This project demonstrates clean code structure, containerized inference, health probes, and CI/CD automation — all in a minimal, easy-to-understand example.

---

## 🚀 Features
- FastAPI microservice with /health and /recommend/{user_id}
- Dockerized Python 3.11-slim runtime
- Kubernetes Deployment + Service with:
  - Readiness & Liveness Probes
  - Resource Requests & Limits
- GitHub Actions CI workflow
- Clean folder structure and simple extensibility

---

## 🧪 Quick Local Run

### 1️⃣ Build & Run Docker
docker build -t iikram42/music-recommendation-system:latest .
docker run --rm -p 8000:8000 iikram42/music-recommendation-system:latest

### 2️⃣ Test Endpoints
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/recommend/1

---

## ☸️ Kubernetes Deployment
kubectl apply -f k8s/deployment.yaml
kubectl port-forward svc/recommender-service 8080:80

---

## 📜 License
MIT License © 2025 iikram42
