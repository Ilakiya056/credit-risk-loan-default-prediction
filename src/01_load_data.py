
import pandas as pd
from pathlib import Path

# Locate the dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "default_of_credit_card_clients.xls"

# Load the Excel dataset
df = pd.read_excel(DATA_PATH, header=1)

# Display basic information
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum().sum())

