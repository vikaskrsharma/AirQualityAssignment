import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

filepath = "../data/air_quality_city_day.csv";
df = pd.read_csv(filepath)
logger.info("Dataset loaded successfully.")

"""Preprocess the dataset by handling missing values, encoding, and feature engineering"""
# Print columns with missing values and their count
missing_values = df.isna().sum()
columns_with_missing = missing_values[missing_values > 0]
logger.info("Columns with missing values: ")
logger.info(columns_with_missing)

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

#Use LabelEncoder to encode string values to numeric 
le = LabelEncoder()  
df_normalized['City']=le.fit_transform(df_normalized['City'])
df_normalized['AQI_Bucket']=le.fit_transform(df_normalized['AQI_Bucket'])

# Print the normalized Encoded dataframe
logger.info("Normalize And Encoded DataFrame:")
logger.info(df_normalized.head());

