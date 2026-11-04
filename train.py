import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, classification_report

ARTIFACTS_DIR = os.path.join("app", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def generate_synthetic_credit_card_data(n_samples=50000, fraud_ratio=0.008):
    """Generate synthetic transaction data replicating severe class imbalance (<1% fraud)"""
    print(f"Generating synthetic dataset  with {n_samples}  samples (Fraud Ratio: {fraud_ratio *100:.2f}%)...")

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    X, y = make_classification(
        n_samples=n_samples,
        n_features=30, # Amount + Time + V1...V28
        n_informative=24,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.001,
        random_state=42
    )

    feature_names = ["amount", "time_delta"] + [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df 

def train_fraud_pipeline():
    # 1. Load Data
    df = generate_synthetic_credit_card_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Train shapes: {X_train.shape} |  Test shapes: {X_test.shape}")
    print(f"Train Fraud Count: {sum(y_train)} | Test Fraud Count: {sum(y_test)}")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled =  scaler.transform(X_test)


    print("\n[1/2] Training Isolation Forest for zero-day anomaly detection...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_scaled)

    # 5. Build Supervised SMOTE + XGBoost Pipeline
    print("[2/2] Training Supervised SMOTE + XGBoost Pipeline...")

    # Calculate scale_pos_weight for imbalance adjustment in XGBoost
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)

    # Imblearn Piepline guarantees SMOTE is ONLY executed on training folds
    pipeline = ImbPipeline([
        ('smote', SMOTE(sampling_strategy= 0.1, random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators= 150,
            max_depth=5,
            learning_rate= 0.05,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train_scaled, y_train)

    # 6. Evaluate Model using Precision-Recall AUC (PR-AUC)
    print("\n--- Model Evaluation & PR-AUC Optimization ---")
    y_scores = pipeline.predict_proba(X_test_scaled)[:,1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    print(f"-> Precision-Recall AUC (PR AUC): {pr_auc:.4f}")

    # Calculate optimatl threshold maximizing F0.5-Score (prioritizing Precision over recall to avoid false alarms)

    beta = 0.5 
    f_scores = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall + 1e-10)
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"-> Optimal Decision Threshold (Max F0.5 Score): {best_threshold:.4f}")

    # Evaluate at optimal threshold
    y_pred_opt = (y_scores >= best_threshold).astype(int)
    print("\nClassification Report (at Optimal Threshold):")
    print(classification_report(y_test, y_pred_opt, target_names=["Legitimate", "Fraud"]))

    # 7. Export Serialized Artifacts 
    print("\nExporting model artifacts to app/artifacts/...")

    # Save metadata threshold for production reference 
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "robust_scaler.joblib"))
    joblib.dump(iso_forest, os.path.join(ARTIFACTS_DIR, "iso_forest.joblib"))
    joblib.dump(pipeline.named_steps['xgb'], os.path.join(ARTIFACTS_DIR, "model_meta.joblib"))

    print("All artifacts successfully serialized and saved!")

if __name__ == "__main__":
    train_fraud_pipeline()