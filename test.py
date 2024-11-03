import pandas as pd
import numpy as np

# Load the existing clean dataset
data_clean = pd.read_csv('data/data_clean.csv')

# Determine the number of synthetic samples to generate
num_synthetic_samples = 50  # Adjust this number as needed

# Generate synthetic data using random sampling based on the distribution of existing data
synthetic_data = pd.DataFrame()

# For numerical columns, use normal distribution to generate synthetic data
for column in data_clean.select_dtypes(include=[np.number]).columns:
    mean = data_clean[column].mean()
    std = data_clean[column].std()
    synthetic_data[column] = np.random.normal(mean, std, num_synthetic_samples)

# For categorical columns, use random sampling from the unique values
for column in data_clean.select_dtypes(include=['object']).columns:
    unique_values = data_clean[column].unique()
    synthetic_data[column] = np.random.choice(unique_values, num_synthetic_samples)

# Combine the original data with the synthetic data
data_combined = pd.concat([synthetic_data], ignore_index=True)

# Save the combined dataset to a new CSV file (optional)
data_combined.to_csv('app/data_test.csv', index=False)

# Print the first few rows of the combined dataset
print("Combined Dataset with Synthetic Data:")
print(data_combined.head())
