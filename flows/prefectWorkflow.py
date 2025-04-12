from prefect import flow, task
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Task 1: Data Ingestion
@task
def ingest_data():
    filepath = "../data/air_quality_city_day.csv"
    df = pd.read_csv(filepath)
    print("Data Ingested Successfully!")

# Task 2: Data Preprocessing
@task
def preprocess_data():
    df = pd.read_csv("../data/air_quality_city_day.csv")
    
   # Select only numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

# Fill missing values in numeric columns with the median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Normalize numeric columns
    scaler = MinMaxScaler()
    numeric_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode categorical column
    encoder = LabelEncoder()
    df["City"] = encoder.fit_transform(df["City"])

    df.to_csv("../data/air_quality_cleaned.csv", index=False)
    print("Data Preprocessing Done!")

# Task 3: Exploratory Data Analysis (EDA)
@task
def perform_eda():
    df = pd.read_csv("../data/air_quality_cleaned.csv")

    # Convert date columns to datetime (if they exist)
    date_cols = ["Date"]  # Adjust based on actual column names
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")  # Convert, ignore errors

    # Drop non-numeric columns before correlation matrix
    df_numeric = df.select_dtypes(include=["number"])

    # Correlation Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Air Pollutants")
    # plt.savefig("../output/correlation_matrix.png")
    plt.show()

    # Feature Importance
    X = df[["PM2.5","PM10", "NO2", "SO2", "CO", "O3"]]
    y = df["AQI"]
    model = RandomForestRegressor()
    model.fit(X, y)

    plt.figure(figsize=(6,4))
    sns.barplot(x=model.feature_importances_, y=X.columns)
    plt.title("Feature Importance for PM2.5 Prediction")
    # plt.savefig("../output/feature_importance.png")
    plt.show()

    print("EDA Completed!")

# Prefect Flow
@flow
def data_pipeline():
    ingest_data()
    preprocess_data()
    perform_eda()

# Run the flow
if __name__ == "__main__":
    data_pipeline.serve(name="air-quality-workflow",
                      tags=["first test workflow"],
                      parameters={},
                      interval=60)
