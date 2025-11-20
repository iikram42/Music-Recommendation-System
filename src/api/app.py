from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from typing import List

app = FastAPI(title="Recommender API")

MODEL_PATH = "artifacts/model.joblib"
model_bundle = None

@app.on_event("startup")
def load_model():
    global model_bundle
    model_bundle = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = 5):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    U = model_bundle['U']
    V = model_bundle['V']
    user_map = model_bundle['user_map']
    item_map = model_bundle['item_map']
    if user_id not in user_map:
        raise HTTPException(status_code=404, detail="User not found")
    u_idx = user_map[user_id]
    scores = U[u_idx].dot(V.T)
    top_idx = np.argsort(-scores)[:top_k]
    inv_item_map = {v:k for k,v in item_map.items()}
    recommendations = [inv_item_map[i] for i in top_idx]
    return {"user_id":user_id, "recommendations": recommendations}
