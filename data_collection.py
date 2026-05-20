import yfinance as yf
import pandas as pd

# Forex pair
pair = "EURUSD=X"

# Download data
data = yf.download(
    pair,
    start="2020-01-01",
    end="2025-01-01"
)

# Show first rows
print(data.head())

# Save CSV
data.to_csv("forex_data.csv")

print("\nForex data downloaded successfully!")