# Installations
# 1. pip install mlflow
# 2. pip install psutil
# psutil is a cross-platform library for retrieving information on running processes and system utilization (CPU, memory, disks, network, sensors) in Python

# Steps to run
# 1. In terminal, run command -> mlflow ui --host 0.0.0.0 --port 5000
# 2. Right click on main.py and "run in interactive terminal"
# 3. Open localhost:5000 in browser and see the experimental results

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import psutil
import time

# Set the MLflow tracking URI to 'http'
mlflow.set_tracking_uri("http://localhost:5000")

# Function for data preprocessing
def preprocess_data(data):
    data = data.drop('Date', axis=1)
    # Fill missing numerical values with median
    numerical_columns = data.select_dtypes(include=["number"]).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())
    data = data.dropna(subset=['AQI_Bucket'])
    
    # Convert categorical variables to one-hot encoding
    data = pd.get_dummies(data, columns=['City'])
    print(data.isnull().sum())

    # Split data into X (features) and y (target)
    X = data.drop('AQI_Bucket', axis=1)
    y = data['AQI_Bucket']

    return X, y

# Function for training the model
def train_model(X_train, y_train, max_depth=3, n_estimators=100):
    # Initialize the classifier
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    return clf

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Function to log model and system metrics to MLflow
def log_to_mlflow(model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Log hyper parameters using in Random Forest Algorithm
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("n_estimators", model.n_estimators)

        # Log model metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        confusion = confusion_matrix(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)
        
        # Log confusion matrix
        confusion_dict = {
            "true_positive": confusion[1][1],
            "false_positive": confusion[0][1],
            "true_negative": confusion[0][0],
            "false_negative": confusion[1][0]
        }
        mlflow.log_metrics(confusion_dict)

        # Log system metrics
        # Example: CPU and Memory Usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        mlflow.log_metric("system_cpu_usage", cpu_usage)
        mlflow.log_metric("system_memory_usage", memory_usage)

        # Log execution time for training the model
        execution_time = {}  # Dictionary to store execution times for different stages
        # Example: Execution time for training the model
        start_time = time.time()
        model = train_model(X_train, y_train)
        end_time = time.time()
        execution_time["system_model_training"] = end_time - start_time

        # Log execution time 
        mlflow.log_metrics(execution_time)

        # Evaluate model and log metrics
        evaluate_model(model, X_test, y_test)

        # Log model
        mlflow.sklearn.log_model(model, "model")

# Main function
def main():
    # Load the dataset
    data = pd.read_csv("../data/air_quality_city_day.csv")

    # Preprocess the data
    X, y = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate and log to MLflow
    log_to_mlflow(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()