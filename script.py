# stock_price_predictor_with_moving_averages.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('all_stocks_5yr.csv')
df_cleaned = df.dropna()

df_cleaned['MA_20'] = df_cleaned.groupby('Name')['close'].transform(lambda x: x.rolling(window=20).mean())
df_cleaned['MA_50'] = df_cleaned.groupby('Name')['close'].transform(lambda x: x.rolling(window=50).mean())

df_cleaned = df_cleaned.dropna()

X_ma = df_cleaned[['open', 'high', 'low', 'volume', 'MA_20', 'MA_50']]

y_ma = df_cleaned['close']

# Split the data into training and testing sets (80% train, 20% test)
X_train_ma, X_test_ma, y_train_ma, y_test_ma = train_test_split(X_ma, y_ma, test_size=0.2, random_state=42)


scaler_ma = StandardScaler()
X_train_ma_scaled = scaler_ma.fit_transform(X_train_ma)
X_test_ma_scaled = scaler_ma.transform(X_test_ma)

# Create a Linear Regression model
model_ma = LinearRegression()

# Cross-validate the model using 5-fold cross-validation
cross_val_scores_ma = cross_val_score(model_ma, X_train_ma_scaled, y_train_ma, cv=5, scoring='neg_mean_squared_error')


model_ma.fit(X_train_ma_scaled, y_train_ma)
y_pred_ma = model_ma.predict(X_test_ma_scaled)


mse_test_ma = mean_squared_error(y_test_ma, y_pred_ma)

print(f"Cross-Validation MSE Scores (5-fold): {cross_val_scores_ma}")
print(f"Mean Squared Error on Test Set: {mse_test_ma}")

plt.figure(figsize=(10, 6))
plt.plot(y_test_ma.values, label='Actual Prices', color='blue')
plt.plot(y_pred_ma, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction with Moving Averages')
plt.xlabel('Test Data Points')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
