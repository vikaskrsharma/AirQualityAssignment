import pandas as pd

path = "../data/air_quality_city_day.csv"
df = pd.read_csv(path)

print(df)

print("✅ Data Ingested Successfully!")