# Air Quality Prediction Project 🌫️📊

This is a comprehensive project focused on predicting Air Quality Index (AQI) levels using a full-stack data science approach — including data ingestion, preprocessing, exploratory analysis, machine learning modeling, and API integration.

## Objective

To build an end-to-end pipeline that:

- Predicts AQI levels based on pollutant concentrations.
- Helps environmental agencies and policymakers take proactive actions.
- Automates the data pipeline using Prefect.
- Serves predictions through a simple API.

---

## 1. 📊 Data Pipeline

### 1.1 Business Understanding

Poor air quality can lead to severe health and environmental issues. This project aims to predict AQI using real-time pollutant concentrations to inform timely interventions.

### 1.2 Data Ingestion

- Dataset sourced from a public domain (e.g., Kaggle’s Air Quality Dataset).
- Loaded into a Pandas DataFrame for processing.

### 1.3 Data Pre-processing

- Displayed summary statistics to understand dataset structure.
- Handled missing values with appropriate imputation techniques.
- Converted date columns to `datetime` format.
- Normalized pollutant concentrations.
- Encoded categorical variables where necessary.

### 1.4 Exploratory Data Analysis (EDA)

- Computed correlations between various pollutants and AQI.
- Visualized data using scatter plots, histograms, and heatmaps.
- Identified top features influencing AQI levels.

### 1.5 DataOps: Workflow Automation

- Automated using [**Prefect**](https://www.prefect.io/).
- Pipeline scheduled to run every 3 minutes to simulate real-time ingestion and processing.

---

## 2. 🤖 Machine Learning Pipeline

### 2.1 Model Selection

Two supervised regression models were implemented:

- **Random Forest Regressor**: Handles non-linearity and missing data well.
- **Gradient Boosting Regressor**: Excellent for tabular datasets and minimizing prediction error.

Models are trained, evaluated, and logged using **MLflow** for experiment tracking and comparison.

---

## 3. 🌐 API Access

An API is provided to:

- Query AQI predictions by sending pollutant concentration inputs.
- Return model predictions in JSON format.
- Can be extended to integrate with front-end dashboards or alert systems.

---

## 🔧 Tech Stack

- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **Prefect** for pipeline automation
- **MLflow** for experiment tracking
- **FastAPI** or **Flask** (API endpoint for prediction)
- **Docker** (optional for containerization)

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/vikaskrsharma/AirQualityAssignment.git
   ```
2. To run Data Analysis Flows, :
   ```bash
   cd AirQualityDataScience
   cd flows
   python prefectWorkflowFinal
   ```
3. To run Machine Learning Flow:
   ```bash
   cd AirQualityDataScience
   cd mlflow
   python ml_main.py
   ```
4. To get Analysis flow run details using API access, update corresponding prefect account keys:
   ```bash
   cd AirQualityDataScience
   cd api
   python api.py
   python deploymentAPI.py
   python flowAPI.py
   ```
