import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("forex_data.csv", skiprows=2)

# Rename columns
data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Keep only Close prices
close_data = data[['Close']]

# Convert to numeric
close_data['Close'] = pd.to_numeric(close_data['Close'])

# Normalize values between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(close_data)

# Show first 5 rows
print("\nScaled Data:\n")
print(scaled_data[:5])