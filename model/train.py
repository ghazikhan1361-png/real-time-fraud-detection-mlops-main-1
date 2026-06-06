import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                              roc_auc_score, f1_score,
                              precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection")

print("Loading data...")
df = pd.read_csv("data/creditcard.csv")

X = df.drop(["Class", "Time"], axis=1)
y = df["Class"]

scaler = StandardScaler()
X = X.copy()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

with mlflow.start_run(run_name="RandomForest_FraudDetection"):
    n_estimators = 100
    max_depth = 10
    class_weight = "balanced"

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("class_weight", class_weight)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("dataset", "creditcard.csv")
    mlflow.log_param("total_samples", len(df))
    mlflow.log_param("fraud_samples", int(y.sum()))

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    print(f"\nROC-AUC:   {roc_auc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/fraud_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    mlflow.sklearn.log_model(model, "random_forest_model",
                              registered_model_name="FraudDetectionModel")

    print("Model saved: model/fraud_model.pkl")
    print("Scaler saved: model/scaler.pkl")
    print("Model registered in MLflow!")

print("Training complete!")
