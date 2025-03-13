import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


# Load cleaned data
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
plt.show()

# Feature Importance using RandomForest
X = df[["PM2.5","PM10", "NO2", "SO2", "CO", "O3"]]
y = df["AQI"]
model = RandomForestRegressor()
model.fit(X, y)

# Plot Feature Importance
plt.figure(figsize=(6,4))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance for PM2.5 Prediction")
plt.show()

print("✅ EDA Completed!")
