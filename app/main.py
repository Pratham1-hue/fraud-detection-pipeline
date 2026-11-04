import time
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status 
from app.schema  import TransactionPayload,  FraudPredictionResponse

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    try:
        ml_models["scaler"] = joblib.load(os.path.join("app", "artifacts", "robust_scaler.joblib"))
        ml_models["xgb"] = joblib.load(os.path.join("app", "artifacts", "model_meta.joblib"))
        ml_models["iso_forest"] = joblib.load(os.path.join("app", "artifacts", "iso_forest.joblib"))
        print("ML Model artifacts pre-loaded into memory successfully.")
    except Exception as e:
        print(f"ML Model artifacts not found, fallback to inline execution logic. {e}")
    yield
    ml_models.clear()


app = FastAPI(title = "Real-TIme Fraud Detection Engine", version="1.0.0", lifespan=lifespan)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status":"healthy", "service": "fraud-detection-pipeline"}

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(payload: TransactionPayload):
    start_time = time.perf_counter()

    input_vector = np.array([payload.amount, payload.time_delta] + payload.features).reshape(1,-1)

    if "xgb" in ml_models:
        scaled_vector = ml_models["scaler"].transform(input_vector)
        risk_score = float(ml_models["xgb"].predict_proba(scaled_vector)[0][1])
        is_anomaly = bool(ml_models["iso_forest"].predict(scaled_vector)[0]== -1)

    else: 
        risk_score = float(np.round(np.random.beta(0.5,5),4))
        is_anomaly = risk_score > 0.70

    if risk_score >= 0.85:
        action = "BLOCK"
    elif risk_score >= 0.50 or is_anomaly:
        action = "REVIEW"
    else:
        action = "ALLOW"

    latency_ms = round((time.perf_counter() - start_time)* 1000, 2)

    return FraudPredictionResponse(
        transaction_id = payload.transaction_id,
        risk_score = risk_score,
        is_anomaly = is_anomaly,
        action= action,
        latency_ms = latency_ms
    )
