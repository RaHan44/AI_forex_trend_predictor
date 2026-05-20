import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# =========================
# LOAD DATA
# =========================

data = pd.read_csv("forex_data.csv", skiprows=2)

data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

close_data = data[['Close']]

close_data['Close'] = pd.to_numeric(close_data['Close'])

# =========================
# SCALE DATA
# =========================

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(close_data)

# =========================
# CREATE TRAINING DATA
# =========================

x_train = []
y_train = []

for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape for LSTM
x_train = np.reshape(
    x_train,
    (x_train.shape[0], x_train.shape[1], 1)
)

# =========================
# BUILD LSTM MODEL
# =========================

model = Sequential()

# First LSTM Layer
model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(x_train.shape[1], 1)
    )
)

# Second LSTM Layer
model.add(
    LSTM(
        units=50,
        return_sequences=False
    )
)

# Dense Layers
model.add(Dense(units=25))
model.add(Dense(units=1))

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# =========================
# TRAIN MODEL
# =========================

print("\nTraining AI Model...\n")

model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5
)

# =========================
# SAVE MODEL
# =========================

model.save("forex_lstm_model.h5")

print("\nModel trained and saved successfully!")