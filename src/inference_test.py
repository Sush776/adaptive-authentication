import joblib
import pandas as pd

# Load models
rf_model = joblib.load("models/RandomForest.pkl")
xgb_model = joblib.load("models/XGBoost.pkl")
lr_model = joblib.load("models/LogisticRegression.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Sample login event
sample = pd.DataFrame([{
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
}])


# preprocess
processed = preprocessor.transform(sample)

# inference
print("RF Risk:", rf_model.predict_proba(processed)[0][1])
print("XGB Risk:", xgb_model.predict_proba(processed)[0][1])
print("LR Risk:", lr_model.predict_proba(processed)[0][1])
