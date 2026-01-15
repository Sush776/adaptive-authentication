import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load Test Data
# -------------------------------
X_test = pd.read_csv("data/X_test.csv", index_col=0)  # your test features
y_test = pd.read_csv("data/y_test.csv", index_col=0).values.ravel()

# -------------------------------
# Load Preprocessor & Models
# -------------------------------
preprocessor = joblib.load("models/preprocessor.pkl")
rf_model = joblib.load("models/RandomForest.pkl")
xgb_model = joblib.load("models/XGBoost.pkl")
lr_model = joblib.load("models/LogisticRegression.pkl")

# -------------------------------
# Preprocess Test Data
# -------------------------------
X_test_processed = preprocessor.transform(X_test)

# -------------------------------
# Define Adaptive MFA Function
# -------------------------------
def adaptive_mfa(prob):
    if prob < 0.3:
        return "Low Risk → Password only"
    elif prob < 0.7:
        return "Medium Risk → OTP / Email"
    else:
        return "High Risk → Step-up / Biometric MFA"

# -------------------------------
# Models Dictionary
# -------------------------------
models = {"RandomForest": rf_model, "XGBoost": xgb_model, "LogisticRegression": lr_model}

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Store results
all_results = pd.DataFrame(index=X_test.index)

# -------------------------------
# Evaluate Each Model
# -------------------------------
for name, model in models.items():
    print(f"\n--- Evaluating {name} ---")
    
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # Save risk probabilities & MFA decisions
    all_results[name[:2] + "_risk_prob"] = y_proba
    all_results[name[:2] + "_MFA"] = [adaptive_mfa(p) for p in y_proba]
    
    # Feature Importance (if available)
    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])
        
        # Get feature names from preprocessor
        num_cols = preprocessor.named_transformers_['num'].get_feature_names_out() if hasattr(preprocessor.named_transformers_['num'], 'get_feature_names_out') else X_test.select_dtypes(include=np.number).columns
        cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out() if 'cat' in preprocessor.named_transformers_ else []
        binary_cols = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []
        
        all_features = np.concatenate([num_cols, binary_cols, cat_cols])
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10,6))
        plt.title(f"{name} Feature Importance")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), all_features[indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"plots/feature_importance_{name}.png")
        plt.show()

# -------------------------------
# Save Risk + MFA Summary
# -------------------------------
all_results.to_csv("data/model_risk_summary.csv")
print("\nRisk + MFA summary saved to model_risk_summary.csv")

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# Prepare a list to store metrics
metrics_list = []

for name, model in models.items():
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:,1]

    # Compute metrics
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1_score": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    }

    metrics_list.append(metrics)

# Convert to DataFrame and save
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("data\model_performance_metrics.csv", index=False)
print("Saved overall model metrics to model_performance_metrics.csv")

