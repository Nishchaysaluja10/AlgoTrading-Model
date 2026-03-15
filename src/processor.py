import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, target_col='close', volume_col='volume', time_col='timestamp'):
        self.target_col = target_col
        self.volume_col = volume_col
        self.time_col = time_col

    def engineer_features(self, df):
        print("📊 Calculating statistical & temporal features...")
        df = df.copy()
        
        if self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors='coerce')
            df['hour'] = df[self.time_col].dt.hour
            df['minute'] = df[self.time_col].dt.minute
            df['day_of_week'] = df[self.time_col].dt.dayofweek
            df = df.drop(columns=[self.time_col])

        windows = [10, 30] 
        for w in windows:
            rolling_window = df[self.target_col].rolling(window=w)
            df[f'sma_{w}'] = rolling_window.mean()
            df[f'sd_{w}'] = rolling_window.std()
            df[f'z_score_{w}'] = (df[self.target_col] - df[f'sma_{w}']) / (df[f'sd_{w}'] + 1e-8)
            df[f'bb_upper_{w}'] = df[f'sma_{w}'] + (2 * df[f'sd_{w}'])
            df[f'bb_lower_{w}'] = df[f'sma_{w}'] - (2 * df[f'sd_{w}'])
            df[f'bb_width_{w}'] = (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}']) / (df[f'sma_{w}'] + 1e-8)

        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['candle_body'] = df['close'] - df['open']
            df['candle_range'] = df['high'] - df['low']
            df['body_to_range'] = df['candle_body'] / (df['candle_range'] + 1e-8)

        if self.volume_col in df.columns:
            df['vol_sma_10'] = df[self.volume_col].rolling(window=10).mean()
            df['rvol_10'] = df[self.volume_col] / (df['vol_sma_10'] + 1e-8)
        df['return_1m'] = df[self.target_col].pct_change()
        df['return_5m'] = df[self.target_col].pct_change(periods=5)
        df['target_up'] = (df[self.target_col].shift(-1) > df[self.target_col]).astype(int)
        
        # --- SANITIZATION (The Fix) ---
        df = df.replace([np.inf, -np.inf], 0)
        df = df.dropna().reset_index(drop=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(lower=-1e9, upper=1e9)
        
        print(f"✅ Generated {len(df.columns) - 1} features and sanitized data.")
        return df