import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from prefect import flow, task
from datetime import datetime

# Define Prefect tasks for modular processing

@task
def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file"""
    df = pd.read_csv(filepath)
    return df

@task
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset by handling missing values, encoding, and feature engineering"""
    # Convert Date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Fill missing numerical values with median
    numerical_columns = df.select_dtypes(include=["number"]).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

    # Fill missing categorical values with mode
    df["City"] = df["City"].fillna(df["City"].mode()[0])
    df["AQI_Bucket"] = df["AQI_Bucket"].fillna(df["AQI_Bucket"].mode()[0])

    # Encode AQI_Bucket (Label Encoding)
    label_encoder = LabelEncoder()
    df["AQI_Bucket_Encoded"] = label_encoder.fit_transform(df["AQI_Bucket"])

    return df

@task
def perform_eda(df: pd.DataFrame):
    """Perform exploratory data analysis (EDA)"""

    print("Summary Statistics:\n", df.describe())

    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=["number"])

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

@task
def split_data(df: pd.DataFrame):
    """Split data into training and testing sets"""
    features = df.drop(columns=["City", "Date", "AQI", "AQI_Bucket"])
    target = df["AQI"]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

@task
def train_models(X_train, y_train):
    """Train Random Forest and Gradient Boosting models"""
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)

    return rf_model, gb_model

@task
def evaluate_models(models, X_test, y_test):
    """Evaluate trained models using MAE, MSE, and R2 Score"""
    rf_model, gb_model = models
    for model, name in zip([rf_model, gb_model], ["Random Forest", "Gradient Boosting"]):
        y_pred = model.predict(X_test)
        print(f"\n{name} Performance:")
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))

@flow
def air_quality_pipeline(filepath: str):
    """Prefect flow to execute the full pipeline"""
    df = load_data(filepath)
    df = preprocess_data(df)
    perform_eda(df)
    X_train, X_test, y_train, y_test = split_data(df)
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

# Run the pipeline with the dataset file
if __name__ == "__main__":
    air_quality_pipeline("./data/air_quality_city_day.csv")


# FastAPI for API Access
# app = FastAPI()

# @app.get("/get_pipeline_details")
# def get_aqi_pipeline_details():
#     return {"flow_name": "Air Quality Pipeline", "status": "Running", "last_run": time.ctime()}

# if __name__ == "__main__":
    # air_quality_pipeline.serve(name="aqi-details-workflow",
    #                   tags=["aqi workflow"],
    #                   parameters={},
    #                   interval=180) #3 minutes
    # air_quality_pipeline()
