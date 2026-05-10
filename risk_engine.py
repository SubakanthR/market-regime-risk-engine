from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
model   = joblib.load("model/regime_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")
app = FastAPI(title="Market Regime Risk Engine")
class MarketInput(BaseModel):
    RSI:        float
    Volatility: float
    ATR:        float
    MACD:       float
@app.get("/")
def root():
    return {"status": "Risk Engine is running "}
@app.post("/predict")
def predict(data: MarketInput):
    features = np.array([[data.RSI, data.Volatility, data.ATR, data.MACD]])

    proba  = model.predict_proba(features)[0]
    labels = encoder.classes_

    risk_scores = {label: round(float(prob), 4) for label, prob in zip(labels, proba)}
    regime      = labels[np.argmax(proba)]
    crisis_score = risk_scores.get("Crisis", 0.0)

    return {
        "predicted_regime":  regime,
        "crisis_risk_score": crisis_score,
        "probabilities":     risk_scores
    }