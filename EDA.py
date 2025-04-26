import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
# Using a chunksize to handle the large file size
df = pd.read_csv('combinedAirfoilDataLabeled.csv')

# Display basic information about the dataset
print("\n=== Dataset Info ===")
print(df.info())

# Display the first few rows
print("\n=== First Few Rows ===")
print(df.head())

# Display basic statistics
print("\n=== Basic Statistics ===")
print(df.describe())

# Display column names
print("\n=== Columns ===")
print(df.columns.tolist())

# Memory usage
print("\n=== Memory Usage ===")
print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

# Check for total number of NaN values in the entire dataset
total_nan = df.isna().sum().sum()
print("\n=== Total NaN Values ===")
print(f"Total NaN values in dataset: {total_nan}")

# Check NaN values per column
print("\n=== NaN Values Per Column ===")
nan_counts = df.isna().sum()
nan_percentages = (df.isna().sum() / len(df)) * 100

# Create a summary of NaN values
nan_summary = pd.DataFrame({
    'NaN Count': nan_counts,
    'NaN Percentage': nan_percentages
})

# Only show columns that have NaN values
nan_summary_with_nans = nan_summary[nan_summary['NaN Count'] > 0]
print(nan_summary_with_nans)

# If there are any NaN values, show a few example rows containing NaN
if total_nan > 0:
    print("\n=== Example Rows with NaN Values ===")
    print(df[df.isna().any(axis=1)].head())
