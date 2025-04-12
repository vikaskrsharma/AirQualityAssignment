import pandas as pd
import logging

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

filepath = "../data/air_quality_city_day.csv";
df = pd.read_csv(filepath)
logger.info("Dataset loaded successfully.")
 # Convert date columns to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

# Summary statistics
logger.info("Summary Statistics:")
logger.info(f"\n{df.describe(include='all')}")

# Checking for missing values
logger.info("Missing Values:")
logger.info(f"\n{df.isnull().sum()}")

# Data type information
logger.info("Data Types:")
logger.info(f"\n{df.dtypes}")