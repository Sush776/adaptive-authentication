# -------------------------------
# Multi-Model Training & Risk Pipeline
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import os

# Create data folder if it doesn't exist
os.makedirs('../data', exist_ok=True) 

os.makedirs('../notebooks/plots', exist_ok=True)  



# -------------------------------
# Step 0: Load Dataset
# -------------------------------
#df = pd.read_csv('data/synthetic_auth_data_50000.csv')
df = pd.read_csv('../data/synthetic_auth_data_50000.csv')


behavioral_features = ['keystroke', 'mouse_speed']
contextual_numerical = ['failed_attempts_last_24h']
binary_features = ['new_device', 'new_location', 'risky_ip']
categorical_features = ['device', 'browser']
target = 'label'

# Clip outliers for mouse_speed
low, high = df['mouse_speed'].quantile([0.005, 0.995])
df['mouse_speed'] = df['mouse_speed'].clip(lower=low, upper=high)

# -------------------------------
# Step 1: Train/Test Split
# -------------------------------
X = df[behavioral_features + contextual_numerical + binary_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 2: Preprocessing Pipeline
# -------------------------------
num_features = behavioral_features + contextual_numerical
binary_features_pipeline = 'passthrough'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('binary', binary_features_pipeline, binary_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_test.to_csv('../data/X_test.csv')
y_test.to_csv('../data/y_test.csv')
# Save the preprocessor
joblib.dump(preprocessor, '../models/preprocessor.pkl')

# -------------------------------
# Step 3: Initialize Models
# -------------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=(len(y_train[y_train==0])/len(y_train[y_train==1])), use_label_encoder=False, eval_metric='logloss'),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
}

# -------------------------------
# Step 4: Train, Save, Evaluate & Generate Risk Probabilities
# -------------------------------
all_risk_probs = pd.DataFrame(index=X_test.index)

# Start MLflow experiment
mlflow.set_experiment("Adaptive_MFA_Training")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\nTraining {name}...")

        # Train the model
        model.fit(X_train_processed, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:,1]  # Risk probability
        all_risk_probs[name + '_risk_prob'] = y_proba

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score for {name}: {auc:.4f}")

        # Log model parameters
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        mlflow.log_metric("f1_score", report['1']['f1-score'])

        # Save model as MLflow artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Also save locally
        joblib.dump(model, f'../models/{name}.pkl')


# -------------------------------
# Step 5: Feature Importance Plot
# -------------------------------
def plot_feature_importance(model, model_name, preprocessor):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"No feature importance for {model_name}")
        return

    # Get feature names
    num_cols = behavioral_features + contextual_numerical
    binary_cols = binary_features
    cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = np.concatenate([num_cols, binary_cols, cat_cols])

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.title(f"{model_name} Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), all_features[indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"../notebooks/plots/feature_importance_{model_name}.png")  
    plt.show()

for name, model in models.items():
    plot_feature_importance(model, name, preprocessor)

# -------------------------------
# Step 6: Export Risk Probabilities for MFA
# -------------------------------
all_risk_probs['label'] = y_test
all_risk_probs.to_csv('../data/risk_probabilities.csv', index=False)
print("Risk probabilities saved to risk_probabilities.csv")

