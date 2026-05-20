from streamlit_autorefresh import st_autorefresh
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import ta
import yfinance as yf

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ======================================
# PAGE CONFIG (ONLY ONCE)
# ======================================

st.set_page_config(
    page_title="AI Forex Predictor",
    layout="wide"
)

# ======================================
# AUTO REFRESH
# ======================================

st_autorefresh(
    interval=60000,
    key="forex_refresh"
)

# ======================================
# TITLE
# ======================================

st.title("AI-Powered Forex Trend Predictor")

st.write(
    "Deep Learning Forex Prediction System using LSTM and Cloud Computing"
)

# ======================================
# LIVE STATUS
# ======================================

current_time = datetime.now()

st.success(
    f"Live Forex Market Connected | Last Updated: {current_time.strftime('%H:%M:%S')}"
)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("Forex Settings")

forex_pair = st.sidebar.selectbox(
    "Select Forex Pair",
    (
        "EURUSD",
        "GBPUSD",
        "USDINR",
        "USDJPY"
    )
)

st.write(f"Selected Pair: {forex_pair}")

# ======================================
# DOWNLOAD DATA
# ======================================

data = yf.download(
    forex_pair + "=X",
    start="2020-01-01",
    end="2025-01-01"
)

data.reset_index(inplace=True)

# Flatten columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Keep needed columns
data = data[
    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
]

# Convert numeric columns
numeric_columns = ['Open', 'High', 'Low', 'Close']

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column])

# ======================================
# SHOW DATA
# ======================================

st.subheader("Forex Dataset")

st.dataframe(data.head())

# ======================================
# SIMPLE TREND GRAPH
# ======================================

st.subheader(f"{forex_pair} Trend")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(data['Close'])

ax.set_title(f"{forex_pair} Closing Price")

st.pyplot(fig)

# ======================================
# CANDLESTICK CHART
# ======================================

candlestick_data = data.copy()

candlestick_data['Date'] = pd.to_datetime(
    candlestick_data['Date']
)

candlestick_data.set_index('Date', inplace=True)

candlestick_data = candlestick_data[
    ['Open', 'High', 'Low', 'Close']
]

st.subheader("Professional Candlestick Chart")

fig_candle, axlist = mpf.plot(
    candlestick_data.tail(100),
    type='candle',
    mav=(20,50),
    volume=False,
    style='yahoo',
    returnfig=True
)

st.pyplot(fig_candle)

# ======================================
# LOAD MODEL
# ======================================

model = load_model("forex_lstm_model.h5")

# ======================================
# PREPARE DATA
# ======================================

close_data = data[['Close']]

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(close_data)

x_test = []

for i in range(60, len(scaled_data)):
    x_test.append(scaled_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(
    x_test,
    (x_test.shape[0], x_test.shape[1], 1)
)

# ======================================
# PREDICTIONS
# ======================================

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

# ======================================
# PREDICTION GRAPH
# ======================================

st.subheader("AI Prediction Graph")

fig2, ax2 = plt.subplots(figsize=(12,5))

ax2.plot(
    close_data[60:].values,
    label='Real Price'
)

ax2.plot(
    predictions,
    label='Predicted Price'
)

ax2.legend()

st.pyplot(fig2)

# ======================================
# BUY / SELL SIGNAL
# ======================================

latest_real = float(close_data.iloc[-1].values[0])

latest_prediction = float(predictions[-1][0])

st.subheader("Trading Signal")

if latest_prediction > latest_real:
    st.success("BUY Signal")
else:
    st.error("SELL Signal")

# ======================================
# PRICE DISPLAY
# ======================================

st.write("Current Price:", round(latest_real, 5))

st.write("Predicted Price:", round(latest_prediction, 5))

# ======================================
# RSI
# ======================================

data['RSI'] = ta.momentum.RSIIndicator(
    close=data['Close'],
    window=14
).rsi()

st.subheader("RSI Indicator")

fig_rsi, ax_rsi = plt.subplots(figsize=(12,4))

ax_rsi.plot(data['RSI'], label='RSI')

ax_rsi.axhline(70, color='red', linestyle='--')

ax_rsi.axhline(30, color='green', linestyle='--')

ax_rsi.legend()

st.pyplot(fig_rsi)

latest_rsi = data['RSI'].iloc[-1]

if latest_rsi > 70:
    st.warning("Market may be Overbought")

elif latest_rsi < 30:
    st.success("Market may be Oversold")

else:
    st.info("Market is Neutral")

# ======================================
# MACD
# ======================================

macd = ta.trend.MACD(close=data['Close'])

data['MACD'] = macd.macd()

data['MACD_SIGNAL'] = macd.macd_signal()

st.subheader("MACD Indicator")

fig_macd, ax_macd = plt.subplots(figsize=(12,4))

ax_macd.plot(
    data['MACD'],
    label='MACD'
)

ax_macd.plot(
    data['MACD_SIGNAL'],
    label='Signal Line'
)

ax_macd.legend()

st.pyplot(fig_macd)

# ======================================
# BOLLINGER BANDS
# ======================================

bollinger = ta.volatility.BollingerBands(
    close=data['Close'],
    window=20,
    window_dev=2
)

data['BB_HIGH'] = bollinger.bollinger_hband()

data['BB_LOW'] = bollinger.bollinger_lband()

st.subheader("Bollinger Bands")

fig_bb, ax_bb = plt.subplots(figsize=(12,5))

ax_bb.plot(data['Close'], label='Close Price')

ax_bb.plot(data['BB_HIGH'], label='Upper Band')

ax_bb.plot(data['BB_LOW'], label='Lower Band')

ax_bb.legend()

st.pyplot(fig_bb)

# ======================================
# AI CONFIDENCE METER
# ======================================

st.subheader("AI Confidence Meter")

difference = abs(
    latest_prediction - latest_real
)

confidence = max(
    50,
    100 - (difference * 1000)
)

confidence = round(confidence, 2)

st.progress(int(confidence))

st.write(f"AI Confidence: {confidence}%")

# ======================================
# LIVE METRICS
# ======================================

st.subheader("Live Market Metrics")

col1, col2, col3 = st.columns(3)

price_change = latest_prediction - latest_real

price_change_percent = (
    (price_change / latest_real) * 100
)

with col1:
    st.metric(
        "Current Price",
        round(latest_real, 5)
    )

with col2:
    st.metric(
        "Predicted Price",
        round(latest_prediction, 5)
    )

with col3:
    st.metric(
        "Prediction Change %",
        f"{round(price_change_percent,2)}%"
    )