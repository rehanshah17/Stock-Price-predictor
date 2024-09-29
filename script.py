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

def fetch_stock_data(stock_symbol):
    try:
        stock_symbol = stock_symbol.upper()
        
        stock_data = yf.download(stock_symbol, period="5y")
        if stock_data.empty:
            print(f"No data found for {stock_symbol}. Skipping.")
            return None
        stock_data['Name'] = stock_symbol
        stock_data.dropna(inplace=True)
    
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data.dropna(inplace=True)
    
        return stock_data
    except KeyError:
        print(f"KeyError: No data available for {stock_symbol}.")
        return None
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return None

def run_singlethreaded(stock_symbols):
    stock_data_list = []
    start_time = time.time()
    for symbol in stock_symbols:
        data = fetch_stock_data(symbol)
        if data is not None:
            stock_data_list.append(data)
    end_time = time.time()
    duration = end_time - start_time
    return stock_data_list, duration

def run_multithreaded(stock_symbols):
    stock_data_list = []
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        stock_data_list = list(executor.map(fetch_stock_data, stock_symbols))
    
    end_time = time.time()
    duration = end_time - start_time
    
    stock_data_list = [data for data in stock_data_list if data is not None]
    return stock_data_list, duration

def predict_stock_price(df, model):
    X = df[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50']]
    y = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse_test = mean_squared_error(y_test, y_pred)
    
    future_5_days, future_1_month = forecast_future_prices(df, model, scaler)
    
    return y_test, y_pred, mse_test, future_5_days, future_1_month

def forecast_future_prices(df, model, scaler):
    last_known_data = df.iloc[-1].copy() 
    
    future_5_days = []
    future_1_month = []
    
    for i in range(1, 21):
        next_data = {
            'Open': last_known_data['Open'],
            'High': last_known_data['High'],
            'Low': last_known_data['Low'],
            'Volume': last_known_data['Volume'],
            'MA_20': last_known_data['MA_20'],
            'MA_50': last_known_data['MA_50']
        }
        
        next_features = pd.DataFrame([next_data]) 
        next_features_scaled = scaler.transform(next_features) 
        

        future_price = model.predict(next_features_scaled)[0]
        

        if i <= 5:
            future_5_days.append(future_price)
        future_1_month.append(future_price)
        
        last_known_data['MA_20'] = (last_known_data['MA_20'] * 19 + future_price) / 20  # Rolling MA_20
        last_known_data['MA_50'] = (last_known_data['MA_50'] * 49 + future_price) / 50  # Rolling MA_50
    
    return future_5_days, future_1_month



if __name__ == "__main__":
    stock_symbol = input("Enter stock ticker symbol (e.g., 'AAPL', 'GOOGL'): ").strip()
    
    stock_data = fetch_stock_data(stock_symbol)
    
    if stock_data is not None:
        single_thread_data, single_thread_duration = run_singlethreaded([stock_symbol])

        multi_thread_data, multi_thread_duration = run_multithreaded([stock_symbol])

        # Calculate speedup
        speedup_percentage = ((single_thread_duration - multi_thread_duration) / single_thread_duration) * 100

        model = LinearRegression()

        y_test, y_pred, mse_test, future_5_days, future_1_month = predict_stock_price(stock_data, model)

        print(f"\nMean Squared Error on Test Set for {stock_symbol.upper()}: {mse_test:.6f}")
        
        print("\nForecasted Prices for the Next 5 Days:")
        for i, price in enumerate(future_5_days, start=1):
            print(f"Day {i}: {price:.2f}")
        
        print("\nForecasted Prices for the Next 1 Month:")
        for i, price in enumerate(future_1_month, start=1):
            print(f"Day {i}: {price:.2f}")

        print("\n--- Performance Comparison ---")
        print(f"Single-threaded execution time: {single_thread_duration:.2f} seconds")
        print(f"Multi-threaded execution time: {multi_thread_duration:.2f} seconds")
        print(f"Multi-threading was {speedup_percentage:.2f}% faster than single-threading.")

    else:
        print(f"\nCould not fetch data for {stock_symbol}. Please try again with a valid ticker.")
