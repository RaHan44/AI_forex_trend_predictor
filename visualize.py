import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv("forex_data.csv", skiprows=2)

# Rename columns
data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert Close column to numeric
data['Close'] = pd.to_numeric(data['Close'])

# Show first rows
print(data.head())

# Create graph
plt.figure(figsize=(14,6))

plt.plot(data['Close'], label='EUR/USD Close Price')

# Titles
plt.title("EUR/USD Forex Trend", fontsize=18)
plt.xlabel("Days")
plt.ylabel("Price")

# Grid and legend
plt.grid(True)
plt.legend()

# Show graph
plt.show()