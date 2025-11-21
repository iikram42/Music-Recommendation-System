# Music Recommendation System

A FastAPI-based recommendation microservice packaged with Docker and deployable to Kubernetes.

## Features
- /health endpoint
- /recommend/{user_id} endpoint
- Dockerfile (Python 3.11-slim)
- Kubernetes deployment (readiness/liveness probes)
- GitHub CI/CD workflow
- Clean project structure

## Run Locally
docker build -t iikram42/music-recommendation-system:latest .
docker run --rm -p 8000:8000 iikram42/music-recommendation-system:latest

## Test
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/recommend/1

## Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl port-forward svc/recommender-service 8080:80

