import joblib
import pandas as pd

# -------------------------------
# Step 1: Load models & preprocessor
# -------------------------------
rf_model = joblib.load("models/RandomForest.pkl")
xgb_model = joblib.load("models/XGBoost.pkl")
lr_model = joblib.load("models/LogisticRegression.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# -------------------------------
# Step 2: Define Adaptive MFA Logic
# -------------------------------
def adaptive_mfa(prob):
    if prob < 0.3:
        return "Low Risk → Password only"
    elif prob < 0.7:
        return "Medium Risk → OTP / Email"
    else:
        return "High Risk → Step-up / Biometric MFA"

# -------------------------------
# Step 3: Predict function
# -------------------------------
def predict_risk(df):
    """
    Input: df - DataFrame containing one or more login events
    Output: DataFrame with model risk probabilities and MFA decision
    """
    processed = preprocessor.transform(df)
    
    results = pd.DataFrame(index=df.index)
    
    for name, model in {"RF": rf_model, "XGB": xgb_model, "LR": lr_model}.items():
        prob = model.predict_proba(processed)[:,1]  # Risk probability
        results[name + "_risk_prob"] = prob
        results[name + "_MFA"] = [adaptive_mfa(p) for p in prob]
    
    return results

# -------------------------------
# Step 4: Test with sample input
# -------------------------------
if __name__ == "__main__":
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
    
    output = predict_risk(sample)
    print(output)
