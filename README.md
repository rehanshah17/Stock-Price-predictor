# Stock Price Predictor with Moving Averages

This project is a stock price prediction tool that leverages moving averages and linear regression to forecast stock prices. The script fetches historical stock data for the last 5 years using the `yfinance` library and predicts the future stock prices for the next 5 days and 1 month. It also compares the performance of fetching stock data using both single-threaded and multi-threaded approaches.

## Features

- Fetch historical stock data for the last 5 years using `yfinance`.
- Calculate 20-day and 50-day moving averages for each stock.
- Predict future stock prices using Linear Regression.
- Forecast stock prices for the next 5 days and 1 month (20 trading days).
- Compare execution times between single-threaded and multi-threaded data fetching.
- Provide a user-friendly output for forecasted stock prices and performance comparison.

