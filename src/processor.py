import pandas as pd
import numpy as np

class DataProcessor:
    """
    Feature engineering for OHLCV data.
    Handles both capitalized (Open, High, Low, Close, Volume) and 
    lowercase (open, high, low, close, volume) column names.
    """
    def __init__(self, target_col='close', volume_col='volume'):
        self.target_col = target_col
        self.volume_col = volume_col

    def _normalize_columns(self, df):
        """Normalize column names to lowercase for consistency."""
        col_map = {col: col.lower() for col in df.columns}
        df = df.rename(columns=col_map)
        return df

    def engineer_features(self, df, training=True):
        print("📊 Calculating statistical & temporal features...")
        df = df.copy()
        
        # Normalize column names to lowercase
        df = self._normalize_columns(df)
        
        close = self.target_col  # 'close'
        
        # ================================================================
        # ROLLING WINDOW FEATURES (SMA, StdDev, Z-Score, Bollinger Bands)
        # ================================================================
        windows = [3, 5, 10, 30] 
        for w in windows:
            rolling_window = df[close].rolling(window=w)
            df[f'sma_{w}'] = rolling_window.mean()
            df[f'sd_{w}'] = rolling_window.std()
            df[f'z_score_{w}'] = (df[close] - df[f'sma_{w}']) / (df[f'sd_{w}'] + 1e-8)
            df[f'bb_upper_{w}'] = df[f'sma_{w}'] + (2 * df[f'sd_{w}'])
            df[f'bb_lower_{w}'] = df[f'sma_{w}'] - (2 * df[f'sd_{w}'])
            df[f'bb_width_{w}'] = (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}']) / (df[f'sma_{w}'] + 1e-8)
            # Normalized distance from price to Bollinger Bands (0 = lower band, 1 = upper band)
            df[f'bb_pos_{w}'] = (df[close] - df[f'bb_lower_{w}']) / (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}'] + 1e-8)

        # ================================================================
        # RSI (Relative Strength Index)
        # ================================================================
        delta = df[close].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # ================================================================
        # MACD (Moving Average Convergence Divergence)
        # ================================================================
        ema_12 = df[close].ewm(span=12, adjust=False).mean()
        ema_26 = df[close].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ================================================================
        # ATR (Average True Range) — Volatility via High/Low/Close
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
            # Upper and lower shadows (wick analysis)
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        # ================================================================
        # STOCHASTIC OSCILLATOR (%K / %D)
        # ================================================================
        stoch_period = 14
        df['stoch_k'] = ((df[close] - df[close].rolling(stoch_period).min()) /
                         (df[close].rolling(stoch_period).max() - 
                          df[close].rolling(stoch_period).min() + 1e-8)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ================================================================
        # ROC (Rate of Change) — 10-period momentum
        # ================================================================
        df['roc_10'] = df[close].pct_change(periods=10) * 100

        # ================================================================
        # VOLUME FEATURES
        # ================================================================
        if self.volume_col in df.columns:
            df['vol_sma_10'] = df[self.volume_col].rolling(window=10).mean()
            df['rvol_10'] = df[self.volume_col] / (df['vol_sma_10'] + 1e-8)
            # VWAP deviation (volume-weighted average price)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                cum_tp_vol = (typical_price * df[self.volume_col]).rolling(window=20).sum()
                cum_vol = df[self.volume_col].rolling(window=20).sum()
                df['vwap_20'] = cum_tp_vol / (cum_vol + 1e-8)
                df['vwap_dev'] = (df['close'] - df['vwap_20']) / (df['vwap_20'] + 1e-8)
        
        # ================================================================
        # MOMENTUM & LAGS
        # ================================================================
        df['return_1m'] = df[close].pct_change()
        df['return_5m'] = df[close].pct_change(periods=5)
        
        df['lag_return_1'] = df['return_1m'].shift(1)
        df['lag_return_2'] = df['return_1m'].shift(2)
        df['lag_return_3'] = df['return_1m'].shift(3)
        
        # Volatility of returns
        df['volatility_5'] = df['return_1m'].rolling(window=5).std()
        
        # Return acceleration (momentum of momentum)
        df['return_accel'] = df['return_1m'].diff()
        
        # ================================================================
        # TARGET & CLEANUP
        # ================================================================
        if training:
            df['target_up'] = (df[close].shift(-1) > df[close]).astype(int)
            df = df.dropna().reset_index(drop=True)
        else:
            # For inference, drop rows where ANY numeric feature is NaN
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df.dropna(subset=numeric_cols).reset_index(drop=True)
        
        # --- SANITIZATION ---
        df = df.replace([np.inf, -np.inf], 0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(lower=-1e9, upper=1e9)
        
        print(f"✅ Generated features. Mode: {'Training' if training else 'Inference'}")
        return df