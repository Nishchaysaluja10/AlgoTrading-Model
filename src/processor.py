import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, target_col='close', volume_col='volume', time_col='timestamp'):
        self.target_col = target_col
        self.volume_col = volume_col
        self.time_col = time_col

    def engineer_features(self, df, training=True):
        print("📊 Calculating statistical & temporal features...")
        df = df.copy()
        
        if self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors='coerce')
            df['hour'] = df[self.time_col].dt.hour
            df['minute'] = df[self.time_col].dt.minute
            df['day_of_week'] = df[self.time_col].dt.dayofweek

        # ================================================================
        # ROLLING WINDOW FEATURES (SMA, StdDev, Z-Score, Bollinger Bands)
        # ================================================================
        windows = [3, 5, 10, 30] 
        for w in windows:
            rolling_window = df[self.target_col].rolling(window=w)
            df[f'sma_{w}'] = rolling_window.mean()
            df[f'sd_{w}'] = rolling_window.std()
            df[f'z_score_{w}'] = (df[self.target_col] - df[f'sma_{w}']) / (df[f'sd_{w}'] + 1e-8)
            df[f'bb_upper_{w}'] = df[f'sma_{w}'] + (2 * df[f'sd_{w}'])
            df[f'bb_lower_{w}'] = df[f'sma_{w}'] - (2 * df[f'sd_{w}'])
            df[f'bb_width_{w}'] = (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}']) / (df[f'sma_{w}'] + 1e-8)
            # NEW: Normalized distance from price to Bollinger Bands
            df[f'bb_pos_{w}'] = (df[self.target_col] - df[f'bb_lower_{w}']) / (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}'] + 1e-8)

        # ================================================================
        # RSI (Relative Strength Index)
        # ================================================================
        delta = df[self.target_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # ================================================================
        # MACD (Moving Average Convergence Divergence)
        # ================================================================
        ema_12 = df[self.target_col].ewm(span=12, adjust=False).mean()
        ema_26 = df[self.target_col].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ================================================================
        # NEW: ATR (Average True Range) — Volatility via High/Low/Close
        # ================================================================
        if all(col in df.columns for col in ['high', 'low', 'close']):
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(window=14).mean()
            # Normalized ATR (relative to price level)
            df['atr_pct'] = df['atr_14'] / (df['close'] + 1e-8)

        # ================================================================
        # CANDLE PATTERNS
        # ================================================================
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['candle_body'] = df['close'] - df['open']
            df['candle_range'] = df['high'] - df['low']
            df['body_to_range'] = df['candle_body'] / (df['candle_range'] + 1e-8)
            # NEW: Upper and lower shadows (wick analysis)
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        # ================================================================
        # NEW: STOCHASTIC OSCILLATOR (%K / %D)
        # ================================================================
        stoch_period = 14
        df['stoch_k'] = ((df[self.target_col] - df[self.target_col].rolling(stoch_period).min()) /
                         (df[self.target_col].rolling(stoch_period).max() - 
                          df[self.target_col].rolling(stoch_period).min() + 1e-8)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ================================================================
        # NEW: ROC (Rate of Change) — 10-period momentum
        # ================================================================
        df['roc_10'] = df[self.target_col].pct_change(periods=10) * 100

        # ================================================================
        # VOLUME FEATURES
        # ================================================================
        if self.volume_col in df.columns:
            df['vol_sma_10'] = df[self.volume_col].rolling(window=10).mean()
            df['rvol_10'] = df[self.volume_col] / (df['vol_sma_10'] + 1e-8)
            # NEW: VWAP deviation (volume-weighted average price)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                cum_tp_vol = (typical_price * df[self.volume_col]).rolling(window=20).sum()
                cum_vol = df[self.volume_col].rolling(window=20).sum()
                df['vwap_20'] = cum_tp_vol / (cum_vol + 1e-8)
                df['vwap_dev'] = (df['close'] - df['vwap_20']) / (df['vwap_20'] + 1e-8)
        
        # ================================================================
        # MOMENTUM & LAGS
        # ================================================================
        df['return_1m'] = df[self.target_col].pct_change()
        df['return_5m'] = df[self.target_col].pct_change(periods=5)
        
        df['lag_return_1'] = df['return_1m'].shift(1)
        df['lag_return_2'] = df['return_1m'].shift(2)
        df['lag_return_3'] = df['return_1m'].shift(3)  # NEW: extra lag
        
        # Volatility of returns
        df['volatility_5'] = df['return_1m'].rolling(window=5).std()
        
        # NEW: Return acceleration (momentum of momentum)
        df['return_accel'] = df['return_1m'].diff()
        
        # ================================================================
        # TARGET & CLEANUP
        # ================================================================
        if training:
            df['target_up'] = (df[self.target_col].shift(-1) > df[self.target_col]).astype(int)
            df = df.dropna().reset_index(drop=True)
        else:
            # For inference, drop rows where ANY numeric feature is NaN
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df.dropna(subset=numeric_cols).reset_index(drop=True)

        if self.time_col in df.columns:
            df = df.drop(columns=[self.time_col])
        
        # --- SANITIZATION ---
        df = df.replace([np.inf, -np.inf], 0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(lower=-1e9, upper=1e9)
        
        print(f"✅ Generated features. Mode: {'Training' if training else 'Inference'}")
        return df