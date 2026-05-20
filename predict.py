import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import load_model

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
# CREATE TEST DATA
# =========================

x_test = []

for i in range(60, len(scaled_data)):
    x_test.append(scaled_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(
    x_test,
    (x_test.shape[0], x_test.shape[1], 1)
)

# =========================
# LOAD TRAINED MODEL
# =========================

model = load_model("forex_lstm_model.h5")

# =========================
# MAKE PREDICTIONS
# =========================

predictions = model.predict(x_test)

# Convert back to original prices
predictions = scaler.inverse_transform(predictions)

# =========================
# VISUALIZATION
# =========================

real_prices = close_data[60:].values

plt.figure(figsize=(14,6))

plt.plot(real_prices, label='Real Price')

plt.plot(predictions, label='Predicted Price')

plt.title("Forex Price Prediction")

plt.xlabel("Time")

plt.ylabel("EUR/USD Price")

plt.legend()

plt.grid(True)

plt.show(block=True)