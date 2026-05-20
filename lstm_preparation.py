import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("forex_data.csv", skiprows=2)

# Rename columns
data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Keep only Close price
close_data = data[['Close']]

# Convert to numeric
close_data['Close'] = pd.to_numeric(close_data['Close'])

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)

# Create training sequences
x_data = []
y_data = []

# Use previous 60 values to predict next value
for i in range(60, len(scaled_data)):
    x_data.append(scaled_data[i-60:i, 0])
    y_data.append(scaled_data[i, 0])

# Convert to numpy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)

# Reshape for LSTM
x_data = np.reshape(
    x_data,
    (x_data.shape[0], x_data.shape[1], 1)
)

# Show shapes
print("X Shape:", x_data.shape)
print("Y Shape:", y_data.shape)