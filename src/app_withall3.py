from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib

# -------------------------------
# Load Models & Preprocessor
# -------------------------------
rf_model = joblib.load("models/RandomForest.pkl")
xgb_model = joblib.load("models/XGBoost.pkl")
lr_model = joblib.load("models/LogisticRegression.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Adaptive Multi-Modal Authentication API")

# -------------------------------
# Input Model
# -------------------------------
class LoginEvent(BaseModel):
    keystroke: float
    mouse_speed: float
    failed_attempts_last_24h: int
    new_device: int
    new_location: int
    risky_ip: int
    device: str
    browser: str
    ip_country: str
    timestamp: Optional[str] = None  # Optional if not used in preprocessing

# -------------------------------
# Adaptive MFA Function
# -------------------------------
def adaptive_mfa(prob):
    if prob < 0.3:
        return "Low Risk → Password only"
    elif prob < 0.7:
        return "Medium Risk → OTP / Email"
    else:
        return "High Risk → Step-up / Biometric MFA"

# -------------------------------
# Prediction Function
# -------------------------------
def predict_risk(df: pd.DataFrame) -> pd.DataFrame:
    processed = preprocessor.transform(df)
    
    results = pd.DataFrame(index=df.index)
    for name, model in {"RF": rf_model, "XGB": xgb_model, "LR": lr_model}.items():
        prob = model.predict_proba(processed)[:,1]
        results[name + "_risk_prob"] = prob
        results[name + "_MFA"] = [adaptive_mfa(p) for p in prob]
    
    return results

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/predict")
def predict_endpoint(events: List[LoginEvent]):
    df = pd.DataFrame([event.dict() for event in events])
    result = predict_risk(df)
    return result.to_dict(orient="records")
