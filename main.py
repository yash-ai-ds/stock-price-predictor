import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def generate_dataset(filename="stock_data.csv", n_rows=1000000):
    """
    Programmatically generate a massive 1,000,000 row stock dataset.
    Uses Vectorized numpy operations for maximum performance.
    """
    print(f"[*] Generating realistic dataset with {n_rows:,} rows (This may take a few seconds)...")
    start_time = time.time()
    
    # 1 million minutes is about ~2 years of minute-by-minute data
    dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="min")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Simulate realistic price movement using a random walk (geometric returns)
    # The return has a very tiny drift and some volatility
    returns = np.random.normal(loc=0.0000005, scale=0.0005, size=n_rows)
    
    # Calculate closing prices
    closing_prices = 150.0 * np.exp(np.cumsum(returns))
    
    # Open price is typically the previous minute's closing price
    opening_prices = np.roll(closing_prices, shift=1)
    opening_prices[0] = closing_prices[0] * np.exp(np.random.normal(0, 0.0005))
    
    # High must be the highest in the period (>= max(open, close))
    high_prices = np.maximum(opening_prices, closing_prices) * (1 + np.abs(np.random.normal(0, 0.0002, n_rows)))
    
    # Low must be the lowest in the period (<= min(open, close))
    low_prices = np.minimum(opening_prices, closing_prices) * (1 - np.abs(np.random.normal(0, 0.0002, n_rows)))
    
    # Volume is semi-random integer
    volumes = np.random.randint(1000, 500000, size=n_rows)
    
    # Create the DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "Open": np.round(opening_prices, 2),
        "High": np.round(high_prices, 2),
        "Low": np.round(low_prices, 2),
        "Close": np.round(closing_prices, 2),
        "Volume": volumes
    })
    
    # Intentionally inject a few missing values in the Volume to demonstrate Data Handling
    missing_idx = np.random.choice(n_rows, size=150, replace=False)
    df.loc[missing_idx, 'Volume'] = np.nan
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"[+] Dataset saved to '{filename}' successfully in {time.time() - start_time:.2f} seconds.")

def load_and_preprocess(filename="stock_data.csv"):
    """
    Load the CSV and handle data preprocessing effectively.
    """
    print(f"\n[*] Loading dataset '{filename}'...")
    start_time = time.time()
    df = pd.read_csv(filename)
    print(f"[+] Loaded {len(df):,} rows in {time.time() - start_time:.2f} seconds.")
    
    print("[*] Handling missing values...")
    # Forward fill missing values then drop remanining just in case
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def train_stock_model(df):
    """
    Train a linear regression model to predict the closing price based on other features.
    """
    print("\n[*] Preparing features (Open, High, Low, Volume) and Target (Close)...")
    
    # Features & Target Selection
    features = ["Open", "High", "Low", "Volume"]
    X = df[features]
    y = df["Close"]
    
    # Train-test split (We use shuffle=False for time-series predictability)
    print("[*] Splitting dataset into Training (80%) and Testing (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    print("[*] Training Linear Regression model (Large Data Expected)...")
    model = LinearRegression()
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"[+] Model trained in {time.time() - start_time:.2f} seconds.")
    
    print("\n[*] Evaluating Model on Test Data...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"    - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"    - R-Squared Score (R2): {r2:.4f}")
    
    return model, X_test, y_test, predictions

def visualize_predictions(df, y_test, predictions, subset_size=200):
    """
    Visualize standard trends showing actual vs predicted outputs.
    Graphing 1M points is unreadable, so we plot a subset of the test data.
    """
    print("\n[*] Generating trend visualization...")
    plt.figure(figsize=(12, 6))
    
    # Plotting only a subset of the final N datapoints
    test_dates_subset = df.iloc[y_test.index]['Date'].tail(subset_size)
    actual_subset = y_test.tail(subset_size)
    pred_subset = predictions[-subset_size:]
    
    test_dates_subset = pd.to_datetime(test_dates_subset)
    
    plt.plot(test_dates_subset, actual_subset, label='Actual Close', color='black', linewidth=2)
    plt.plot(test_dates_subset, pred_subset, label='Predicted Close', color='cyan', linestyle='dashed', linewidth=2)
    
    plt.title(f'Stock Price Prediction (Last {subset_size} Steps)')
    plt.xlabel('Date/Time')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_plot.png')
    print("[+] Visualization saved to 'prediction_plot.png'!")

def predict_future(model, open_p, high_p, low_p, volume_p):
    """
    Utility function for manual future inference inputs
    """
    input_df = pd.DataFrame({
        "Open": [open_p],
        "High": [high_p],
        "Low": [low_p],
        "Volume": [volume_p]
    })
    
    pred = model.predict(input_df)[0]
    print(f"\n[*] Predicting Future Stock Close:")
    print(f"    Input  -> Open: {open_p}, High: {high_p}, Low: {low_p}, Volume: {volume_p:,}")
    print(f"    Output -> Predicted Close Price: {pred:.2f}")
    return pred

def main():
    print("="*60)
    print("      STOCK PRICE PREDICTION PIPELINE (1M ROWS)      ")
    print("="*60)
    
    filename = "stock_data.csv"
    
    # 1. Efficient Dataset Generation
    # -------------------------------------------------------------
    # 💡 TIP: Change 'n_rows' below to generate a larger or smaller dataset!
    # For example: n_rows=500000 for half-size, or 5000000 for a massive 5M dataset!
    # -------------------------------------------------------------
    if not os.path.exists(filename):
        generate_dataset(filename, n_rows=1000000)
    else:
        print(f"[*] Dataset '{filename}' already exists. Skipping generation.")
        
    # 2. Data Processing & Cleansing
    df = load_and_preprocess(filename)
    
    # 3. Model Training
    model, X_test, y_test, predictions = train_stock_model(df)
    
    # 4. Visualization Mapping
    visualize_predictions(df, y_test, predictions)
    
    # 5. Feature Predictions from random manual input based on late metrics
    last_row = df.iloc[-1]
    
    # Simulating the next 'period' where you know high, open, low and need to lock-in close
    fut_open = last_row["Close"] 
    fut_high = fut_open + 0.15
    fut_low  = fut_open - 0.20
    fut_vol  = 125000
    
    predict_future(model, fut_open, fut_high, fut_low, fut_vol)

if __name__ == "__main__":
    main()
