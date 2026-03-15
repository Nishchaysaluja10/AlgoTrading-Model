import pandas as pd
import numpy as np
import os

def generate_hackathon_data(num_rows=5000):
    os.makedirs('data', exist_ok=True)
    print("🧪 Synthesizing volatile OHLCV hackathon market data...")
    
    np.random.seed(42)
    timestamps = pd.date_range(start='2026-01-01', periods=num_rows, freq='1min')
    
    opens = [100.0]
    highs = [100.5]
    lows = [99.5]
    closes = [100.0]
    volumes = [1000]
    
    for i in range(1, num_rows):
        # 1. The new Open is just the previous minute's Close
        current_open = closes[i-1]
        
        # 2. Base Random Noise
        price_change = np.random.normal(0, 0.5)
        
        # 3. Inject Artificial Momentum 
        if i > 5:
            trend = closes[i-1] - closes[i-5]
            price_change += trend * 0.15 
            
        # 4. Inject Mean Reversion 
        if i > 20:
            sma_20 = np.mean(closes[i-20:i])
            deviation = closes[i-1] - sma_20
            if abs(deviation) > 2.5: 
                price_change -= deviation * 0.4 
                
        # 5. Calculate the final Close
        current_close = max(1.0, current_open + price_change)
        
        # 6. Generate realistic Highs and Lows for the candle wick
        current_high = max(current_open, current_close) + abs(np.random.normal(0, 0.2))
        current_low = min(current_open, current_close) - abs(np.random.normal(0, 0.2))
        
        # 7. Volume surges on big moves
        vol = np.random.normal(1000, 200) + (abs(price_change) * 1500)
        
        opens.append(current_open)
        highs.append(current_high)
        lows.append(current_low)
        closes.append(current_close)
        volumes.append(max(100, vol))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    save_path = "data/historical_data.csv"
    df.to_csv(save_path, index=False)
    print(f"✅ Generated {num_rows} rows of OHLCV data at {save_path}")

if __name__ == "__main__":
    generate_hackathon_data()