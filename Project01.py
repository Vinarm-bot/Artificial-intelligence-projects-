import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Download stock data (Apple)
stock = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

# Keep only Close price
stock = stock[['Close']]

# Create prediction column (next day's price)
stock['Prediction'] = stock['Close'].shift(-1)

# Remove last row with NaN
stock.dropna(inplace=True)

# Features and Labels
X = stock[['Close']]
y = stock['Prediction']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Predict tomorrow's price
today_price = [[stock['Close'].iloc[-1]]]
tomorrow = model.predict(today_price)

print("\nToday's Closing Price:", today_price[0][0])
print("Predicted Tomorrow Price:", tomorrow[0])

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
