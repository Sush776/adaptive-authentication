from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import joblib
from src.prediction_logging import log_prediction
from src.metrics import metrics_store

# -------------------------------
# Load Preprocessor and Models
# -------------------------------
preprocessor = joblib.load("models/preprocessor.pkl")
rf_model = joblib.load("models/RandomForest.pkl")
xgb_model = joblib.load("models/XGBoost.pkl")
lr_model = joblib.load("models/LogisticRegression.pkl")

# Load model evaluation metrics
eval_df = pd.read_csv("data/model_performance_metrics.csv")
best_model_name = eval_df.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]

models = {
    "RandomForest": rf_model,
    "XGBoost": xgb_model,
    "LogisticRegression": lr_model
}
best_model = models[best_model_name]

# -------------------------------
# Input / Output Pydantic Models
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
    timestamp: Optional[str] = None

class PredictionResult(BaseModel):
    risk_prob: float = Field(..., example=0.74)
    MFA: str = Field(..., example="High Risk → Step-up / Biometric MFA")

# -------------------------------
# Helper Functions
# “Risk thresholds were empirically selected based on ROC-AUC analysis and practical 
# security considerations. A lower threshold minimizes unnecessary user friction for 
# legitimate logins, while a higher threshold prioritizes security by enforcing stronger 
# authentication for high-risk events. This tiered strategy balances usability and 
# security, consistent with adaptive authentication practices in real-world IAM systems.”
# -------------------------------
def adaptive_mfa(prob: float) -> str:
    if prob < 0.3:
        return "Low Risk → Password only"
    elif prob < 0.7:
        return "Medium Risk → OTP / Email"
    else:
        return "High Risk → Step-up / Biometric MFA"

def predict_event(event: LoginEvent) -> PredictionResult:
    df = pd.DataFrame([event.dict()])
    X_processed = preprocessor.transform(df)
    prob = best_model.predict_proba(X_processed)[:, 1][0]
    return PredictionResult(risk_prob=prob, MFA=adaptive_mfa(prob))

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="Adaptive MFA Risk Prediction API",
    description="Predicts login risk scores and recommends adaptive MFA using the best-performing ML model."
)

@app.post("/predict", response_model=List[PredictionResult],
          description="Submit one or multiple login events and receive risk probabilities + MFA recommendations.")
def predict(events: List[LoginEvent] = Body(..., examples={
    "single": {
        "summary": "Single login event",
        "value": [{
            "keystroke": 0.15,
            "mouse_speed": 0.35,
            "failed_attempts_last_24h": 1,
            "new_device": 1,
            "new_location": 0,
            "risky_ip": 0,
            "device": "android",
            "browser": "chrome",
            "ip_country": "IN",
            "timestamp": "2025-12-28 14:30:00"
        }]
    },
    "multiple": {
        "summary": "Multiple login events",
        "value": [
            {
                "keystroke": 0.15,
                "mouse_speed": 0.35,
                "failed_attempts_last_24h": 1,
                "new_device": 1,
                "new_location": 0,
                "risky_ip": 0,
                "device": "android",
                "browser": "chrome",
                "ip_country": "IN",
                "timestamp": "2025-12-28 14:30:00"
            },
            {
                "keystroke": 0.25,
                "mouse_speed": 0.42,
                "failed_attempts_last_24h": 2,
                "new_device": 0,
                "new_location": 1,
                "risky_ip": 1,
                "device": "ios",
                "browser": "safari",
                "ip_country": "US",
                "timestamp": "2025-12-28 15:10:00"
            }
        ]
    }
})):
    # results = []
    # for e in events:
    #     result = predict_event(e)     # model prediction
    #     log_prediction(e, result.risk_prob, result.MFA)     # log this prediction
    #     results.append(result)        # collect result
    
    # return results
    results = []
    for e in events:
        try:
            result = predict_event(e)
            results.append(result)

            # collect metrics
            metrics_store.record_prediction(
                risk_score=result.risk_prob,
                mfa_required=(result.MFA == "REQUIRED")
            )

            # logging to CSV
            log_prediction(e, result.risk_prob, result.MFA)
        except Exception:
            metrics_store.record_error()
            raise

    return results

@app.get("/metrics")
def get_metrics():
    """
    Returns basic model usage metrics (not Prometheus format yet).
    """
    return metrics_store.get_metrics()
