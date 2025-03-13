import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Task 1: Load Data and Handle Missing Values
@task
def load_and_preprocess_data():
    url = "../data/air_quality_city_day.csv"
    df = pd.read_csv(url)

     # ✅ Convert 'Date' column from string to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Handle missing values in AQI_Bucket
    df["AQI_Bucket"].fillna("Moderate", inplace=True)

    # Encode AQI_Bucket (Target Variable)
    label_encoder = LabelEncoder()
    df["AQI_Bucket"] = label_encoder.fit_transform(df["AQI_Bucket"]) 

    # Select Features and Target
    features = ["PM10", "NO2", "SO2", "CO", "O3", "NH3"]
    df = df[features + ["AQI_Bucket"]].dropna()  # Drop rows with missing numerical values

    return df, label_encoder

# Task 2: Split Data into Train-Test Sets
@task
def split_data(df):
    X = df.drop(columns=["AQI_Bucket"])
    y = df["AQI_Bucket"]
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Task 3: Train a Model (Random Forest or XGBoost)
@task
def train_model(X_train, X_test, y_train, y_test, model_type="random_forest"):
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"✅ {model_type} Model Performance:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - F1 Score: {f1:.4f}")

    return model, accuracy, precision, recall, f1

# Task 4: Log Metrics to MLflow
@task
def log_metrics(model, model_type, accuracy, precision, recall, f1):
    mlflow.set_experiment("AQI Prediction")
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, f"models/{model_type}")

# Prefect Flow to Run the Entire Pipeline
@flow
def ml_pipeline():
    df, label_encoder = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Train and Log Random Forest
    model_rf, acc_rf, prec_rf, rec_rf, f1_rf = train_model(X_train, X_test, y_train, y_test, model_type="random_forest")
    log_metrics(model_rf, "random_forest", acc_rf, prec_rf, rec_rf, f1_rf)

    # Train and Log XGBoost
    model_xgb, acc_xgb, prec_xgb, rec_xgb, f1_xgb = train_model(X_train, X_test, y_train, y_test, model_type="xgboost")
    log_metrics(model_rf, "xgboost", acc_xgb, prec_xgb, rec_xgb, f1_xgb)

    print("✅ ML Pipeline Completed Successfully!")

if __name__ == "__main__":
    ml_pipeline()
