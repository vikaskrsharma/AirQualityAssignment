import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from prefect import flow, task
from datetime import datetime


# Define Prefect tasks for modular processing

@task
def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file"""
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully.")
    # print(f"DataFrame head:\n{df.head()}")
    return df

@task
def data_basic_stat(df: pd.DataFrame):
    # Convert date columns to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

    # Summary statistics
    print("Summary Statistics:")
    print(f"\n{df.describe(include='all')}")

    # Checking for missing values
    print("Missing Values:")
    print(f"\n{df.isnull().sum()}")

    # Data type information
    print("Data Types:")
    print(f"\n{df.dtypes}")

@task
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset by handling missing values, encoding, and feature engineering"""

     # Print columns with missing values and their count
    missing_values = df.isna().sum()
    columns_with_missing = missing_values[missing_values > 0]
    print("Columns with missing values: ")
    print(columns_with_missing)

    # Convert Date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    
    # Fill missing numerical values with median
    numerical_columns = df.select_dtypes(include=["number"]).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

     # Normalize using Min-Max Scaling
    scaler = MinMaxScaler()
    features = df.drop(['City', 'Date', 'AQI', 'AQI_Bucket'], axis=1)  # Exclude the target variable
    df_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    # Add the target variable back to the dataframe
    df_normalized['City'] = df['City']
    df_normalized['Date'] = df['Date']
    df_normalized['AQI'] = df['AQI']
    df_normalized['AQI_Bucket'] = df['AQI_Bucket']

    # Print the normalized dataframe
    print("Normalized DataFrame:")
    print(df_normalized.head());

    #Use LabelEncoder to encode string values to numeric 
    le = LabelEncoder()  
    df_normalized['City']=le.fit_transform(df_normalized['City'])
    df_normalized['AQI_Bucket']=le.fit_transform(df_normalized['AQI_Bucket'])
    print(df_normalized.head())

    # # Fill missing categorical values with mode
    # df["City"] = df["City"].fillna(df["City"].mode()[0])
    # df["AQI_Bucket"] = df["AQI_Bucket"].fillna(df["AQI_Bucket"].mode()[0])

    # # Encode AQI_Bucket (Label Encoding)
    # label_encoder = LabelEncoder()
    # df["AQI_Bucket_Encoded"] = label_encoder.fit_transform(df["AQI_Bucket"])

    return df_normalized

@task
def visualize_analysis(df:pd.DataFrame):

    ## *************** Univariate Analysis *******************************

    # AQI Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['AQI'], bins=30, kde=True)
    plt.title('Distribution of AQI Values')
    plt.xlabel('AQI')
    plt.ylabel('Frequency')
    plt.show()

    # AQI Bucket Distribution
    plt.figure(figsize=(8, 5))
    df['AQI_Bucket'].value_counts().plot(kind='bar')
    plt.title('Distribution of AQI Buckets')
    plt.xlabel('AQI Category')
    plt.ylabel('Count')
    plt.show()

    # Pollutant Concentration Distributions
    pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    plt.figure(figsize=(15, 10))
    for i, pollutant in enumerate(pollutants, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(y=df[pollutant])
        plt.title(f'Distribution of {pollutant}')
    plt.tight_layout()
    plt.show()

    # *************************** Bivariate Analysis ********************
    # AQI vs Pollutants
    plt.figure(figsize=(15, 10))
    for i, pollutant in enumerate(pollutants, 1):
        plt.subplot(2, 3, i)
        sns.scatterplot(x=df[pollutant], y=df['AQI'])
        plt.title(f'AQI vs {pollutant}')
    plt.tight_layout()
    plt.show()

    # AQI Trends Over Time
    # Aggregate by month
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_avg = df.groupby('Month')['AQI'].mean().reset_index()
    monthly_avg['Month'] = monthly_avg['Month'].astype(str)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Month', y='AQI', data=monthly_avg)
    plt.title('Monthly Average AQI Trend')
    plt.xticks(rotation=45)
    plt.show()

    #City-wise AQI Comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='City', y='AQI', data=df)
    plt.title('AQI Distribution Across Cities')
    plt.xticks(rotation=45)
    plt.show()

    #AQI Bucket vs Pollutants
    plt.figure(figsize=(15, 10))
    for i, pollutant in enumerate(pollutants, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='AQI_Bucket', y=pollutant, data=df)
        plt.title(f'{pollutant} by AQI Bucket')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # ******************* Feature Importance with Random Forest ********************




@task
def perform_eda(df: pd.DataFrame):
    """Perform exploratory data analysis (EDA)"""

    # Correlation Matrix - Internally uses Pearson Correlation
    cor = df.corr()

    # Plotting Heatmap
    plt.figure(figsize = (10,6))
    sns.heatmap(cor, annot=True)
    plt.show()

    # print("Summary Statistics:\n", df.describe())

    # Prepare features and target
    X = df.drop(['City', 'Date', 'AQI', 'AQI_Bucket'], axis=1)
    y = df['AQI']

    # Fit Random Forest
    model = RandomForestRegressor()
    model.fit(X, y)

    # Get feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance)
    plt.title('Feature Importance for AQI Prediction')
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
    # data_basic_stat(df)
    df_normalized = preprocess_data(df)
    visualize_analysis(df)
    perform_eda(df_normalized)

    # X_train, X_test, y_train, y_test = split_data(df)
    # models = train_models(X_train, y_train)
    # evaluate_models(models, X_test, y_test)

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
