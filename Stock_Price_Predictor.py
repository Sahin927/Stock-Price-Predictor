import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Historical Stock Data
# -------------------------------
ticker = "AAPL"   # Change to any stock symbol
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Use only the "Close" price
data = data[["Close"]]

# -------------------------------
# 2. Prepare Dataset
# -------------------------------
# Predict 1 day into the future
data["Prediction"] = data["Close"].shift(-1)

# Drop last row (NaN label)
X = np.array(data.drop(["Prediction"], axis=1))[:-1]
y = np.array(data["Prediction"])[:-1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------------
# 3. Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 4. Test Model
# -------------------------------
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

last_close = float(data["Close"].dropna().iloc[-1])
next_day_price = model.predict([[last_close]])[0]

# -------------------------------
# 6. Plot Graph
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

