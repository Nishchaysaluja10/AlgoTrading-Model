import yfinance as yf
import pandas as pd
import os

def download_external_test_data():
    os.makedirs('data', exist_ok=True)
    print("🌐 Fetching external market data (Bitcoin OHLCV)...")
    
    # We'll pull 60 days of hourly data for a clean test
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(period="60d", interval="1h")
    
    df = df.reset_index()
    # Mapping Yahoo Finance columns to your specific architecture names
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Datetime": "timestamp"
    })
    
    # Selecting only the attributes your processor now expects
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    save_path = "data/historical_data.csv"
    df.to_csv(save_path, index=False)
    print(f"✅ Saved {len(df)} rows of REAL Bitcoin data to {save_path}")

if __name__ == "__main__":
    download_external_test_data()