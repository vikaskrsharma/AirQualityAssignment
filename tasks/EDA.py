import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import logging
from datetime import datetime


# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

filepath = "./data/air_quality_city_day.csv";
df = pd.read_csv(filepath)
logger.info("Dataset loaded successfully.")

# Create a deep copy of the DataFrame
df_visualize = df.copy(deep=True)
# Convert Date column to datetime format
df_visualize["Date"] = pd.to_datetime(df_visualize["Date"], errors="coerce")

## *************** Univariate Analysis *******************************

# AQI Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_visualize['AQI'], bins=30, kde=True)
plt.title('Distribution of AQI Values')
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.show()

# AQI Bucket Distribution
plt.figure(figsize=(8, 5))
df_visualize['AQI_Bucket'].value_counts().plot(kind='bar')
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
df_visualize['Month'] = df_visualize['Date'].dt.to_period('M')
monthly_avg = df_visualize.groupby('Month')['AQI'].mean().reset_index()
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



## ********************* Normalize Data And Correlation ********************************
# Create a deep copy of the DataFrame
df_corr = df.copy(deep=True)
# Convert Date column to datetime format
df_corr["Date"] = pd.to_datetime(df_corr["Date"], errors="coerce")


# Fill missing numerical values with median
numerical_columns = df_corr.select_dtypes(include=["number"]).columns
df_corr[numerical_columns] = df_corr[numerical_columns].fillna(df_corr[numerical_columns].median())

# Normalize using Min-Max Scaling
scaler = MinMaxScaler()
features = df_corr.drop(['City', 'Date', 'AQI', 'AQI_Bucket'], axis=1)  # Exclude the target variable
df_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
# Add the target variable back to the dataframe
df_normalized['City'] = df_corr['City']
df_normalized['Date'] = df_corr['Date']
df_normalized['AQI'] = df_corr['AQI']
df_normalized['AQI_Bucket'] = df_corr['AQI_Bucket']

#Use LabelEncoder to encode string values to numeric 
le = LabelEncoder()  
df_normalized['City']=le.fit_transform(df_normalized['City'])
df_normalized['AQI_Bucket']=le.fit_transform(df_normalized['AQI_Bucket'])

# Correlation Matrix - Internally uses Pearson Correlation
cor = df_normalized.corr()

# Plotting Heatmap
plt.figure(figsize = (10,6))
sns.heatmap(cor, annot=True)
plt.show()

# Prepare features and target
X = df_normalized.drop(['City', 'Date', 'AQI', 'AQI_Bucket'], axis=1)
y = df_normalized['AQI']

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


