from pathlib import Path
import pandas as pd

# Automatically locate the CSV file relative to this script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "dementia_patients_health_data.csv"

# Fallback: Check if the file sits in the root project folder instead of a 'data' subfolder
if not DATA_PATH.exists():
    DATA_PATH = SCRIPT_DIR / "dementia_patients_health_data.csv"

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded dataset from: {DATA_PATH}\n")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find the dataset at {DATA_PATH}. "
        "Please make sure 'dementia_patients_health_data.csv' exists in your project folder or 'data' subfolder."
    )

# Dataset shape
print("Dataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(f"Total duplicate rows: {df.duplicated().sum()}")

# Dataset information
print("\nDataset Information:")
df.info()

# Target distribution (checks if 'Dementia' column exists)
print("\nDementia Distribution:")
if "Dementia" in df.columns:
    print(df["Dementia"].value_counts())
else:
    print("Warning: Column 'Dementia' not found. Available columns:", df.columns.tolist())