# Summary statistics
# Handling missing values
# Normalizing numerical columns
# Encoding categorical data

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load data
df = pd.read_csv("../data/air_quality_city_day.csv")

# Display summary statistics
# print(df.describe())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Select only numeric columns
numeric_cols = df.select_dtypes(include=["number"]).columns

# Fill missing values in numeric columns with the median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Normalize numeric columns
scaler = MinMaxScaler()
numeric_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df)

# Encode categorical features (e.g., City)
encoder = LabelEncoder()
df["City"] = encoder.fit_transform(df["City"])
df["AQI_Bucket"] = encoder.fit_transform(df["AQI_Bucket"])

# Save preprocessed data
df.to_csv("../data/air_quality_cleaned.csv", index=False)

print(df)
print("✅ Data Pre-processing Completed!")
