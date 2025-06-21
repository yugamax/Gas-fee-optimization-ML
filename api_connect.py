from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import uvicorn
import os

class features(BaseModel):
    hour: float
    peer_count: float
    unconfirmed_count: float
    high_gas_price_gwei: float
    medium_gas_price_gwei: float
    low_gas_price_gwei: float
    high_priority_fee_gwei: float
    medium_priority_fee_gwei: float
    low_priority_fee_gwei: float

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

scaler = joblib.load("model/scaler.joblib")
model = load_model("model/gasfee.keras")

@app.get("/health")
def health_check():
    return {"status": "its running"}

@app.post("/predict")
def predict_fee_class(features: features):
    x = np.array([[
        features.peer_count,
        features.unconfirmed_count,
        features.high_gas_price_gwei,
        features.medium_gas_price_gwei,
        features.low_gas_price_gwei,
        features.high_priority_fee_gwei,
        features.medium_priority_fee_gwei,
        features.low_priority_fee_gwei,
        features.hour
    ]], dtype=float)

    x_scaled = scaler.transform(x)

    preds = model.predict(x_scaled)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds, axis=1)[0])

    label_map = {0: "Low", 1: "Mid", 2: "High"}

    return {
        "predicted_class": label_map[class_idx],
        "confidence": f"{confidence * 100:.2f}%"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)