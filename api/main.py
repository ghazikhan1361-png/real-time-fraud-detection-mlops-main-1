from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import joblib
import numpy as np
import os
import time

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="MLOps Project - Kaab Abdullah Malik (SAP: 70148009)",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/fraud_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

REQUEST_COUNT = Counter("fraud_api_requests_total", "Total API requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("fraud_api_request_duration_seconds", "Request duration")
FRAUD_DETECTED = Counter("fraud_detected_total", "Total fraud predictions")
LEGIT_DETECTED = Counter("legitimate_detected_total", "Total legitimate predictions")

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def root():
    return {
        "message": "Real-Time Fraud Detection API",
        "student": "Kaab Abdullah Malik",
        "sap_id": "70148009",
        "status": "running"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    amount_scaled = scaler.transform([[transaction.Amount]])[0][0]
    features = [[
        transaction.V1, transaction.V2, transaction.V3, transaction.V4,
        transaction.V5, transaction.V6, transaction.V7, transaction.V8,
        transaction.V9, transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28,
        amount_scaled
    ]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    result = "Fraudulent" if prediction == 1 else "Legitimate"
    duration = time.time() - start
    REQUEST_COUNT.labels("POST", "/predict", "200").inc()
    REQUEST_LATENCY.observe(duration)
    if prediction == 1:
        FRAUD_DETECTED.inc()
    else:
        LEGIT_DETECTED.inc()
    return {
        "prediction": result,
        "fraud_probability": round(float(probability), 4),
        "is_fraud": bool(prediction),
        "response_time_ms": round(duration * 1000, 2)
    }

@app.get("/metrics")
def metrics():
    REQUEST_COUNT.labels("GET", "/metrics", "200").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
