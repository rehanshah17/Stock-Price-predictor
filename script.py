# stock_price_predictor_with_moving_averages.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time

# Function to fetch stock data for the last 5 years
def fetch_stock_data(stock_symbol):
    stock_data = yf.download(stock_symbol, period="5y")
    stock_data['Name'] = stock_symbol
    stock_data.dropna(inplace=True)
    
    # Calculate moving averages
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data.dropna(inplace=True)
    
    return stock_data

# Function to prepare and predict stock price
def predict_stock_price(df):
    X = df[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50']]
    y = df['Close']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear Regression model
    model = LinearRegression()
    
    # Cross-validate the model using 5-fold cross-validation
    cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    
    # Train and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate Mean Squared Error on test set
    mse_test = mean_squared_error(y_test, y_pred)
    
    return y_test, y_pred, mse_test, cross_val_scores

# Function to run the prediction for multiple stocks
def run_multithreaded(stock_symbols):
    stock_data_list = []
    
    # Start measuring time for multi-threading
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        stock_data_list = list(executor.map(fetch_stock_data, stock_symbols))
    
    end_time = time.time()
    multithread_duration = end_time - start_time
    print(f"Multi-threaded execution time: {multithread_duration:.2f} seconds")
    
    return stock_data_list, multithread_duration

# Single-threaded version for comparison
def run_singlethreaded(stock_symbols):
    stock_data_list = []
    
    # Start measuring time for single-threading
    start_time = time.time()
    
    for symbol in stock_symbols:
        stock_data_list.append(fetch_stock_data(symbol))
    
    end_time = time.time()
    singlethread_duration = end_time - start_time
    print(f"Single-threaded execution time: {singlethread_duration:.2f} seconds")
    
    return stock_data_list, singlethread_duration

# Main function
if __name__ == "__main__":
    stock_symbols = ['AAPL']  # Example stock symbols
    
    # Run single-threaded first
    single_thread_data, single_thread_time = run_singlethreaded(stock_symbols)
    
    # Run multi-threaded
    multi_thread_data, multi_thread_time = run_multithreaded(stock_symbols)
    
    # Predict stock prices for the multi-threaded data
    for stock_data in multi_thread_data:
        y_test, y_pred, mse_test, cross_val_scores = predict_stock_price(stock_data)
        
        print(f"Cross-Validation MSE Scores (5-fold): {cross_val_scores}")
        print(f"Mean Squared Error on Test Set for {stock_data['Name'].iloc[0]}: {mse_test}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values, label='Actual Prices', color='blue')
        plt.plot(y_pred, label='Predicted Prices', color='red')
        plt.title(f'Stock Price Prediction for {stock_data["Name"].iloc[0]}')
        plt.xlabel('Test Data Points')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
    
    # Calculate speedup percentage
    speedup = ((single_thread_time - multi_thread_time) / single_thread_time) * 100
    print(f"Multi-threading was {speedup:.2f}% faster than single-threading.")
