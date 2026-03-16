"""
=============================================================================
  AGENT.PY — AlgoTrading Hackathon Live Agent
  5-Layer Safety Framework + ML Ensemble (XGB + CatBoost + LightGBM)
=============================================================================
"""
import os, time, requests, numpy as np, pandas as pd, joblib, sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# ENV CONFIG
# ═══════════════════════════════════════════════════════════════════════════
API_URL = os.getenv("API_URL", "http://SERVER_IP:8001")
API_KEY = os.getenv("TEAM_API_KEY", "YOUR_KEY_HERE")
HEADERS = {"X-API-Key": API_KEY}

# ── Thresholds (tuned for cash decay vs fee tradeoff) ──
# prob > BUY_PROB  → model says price will go UP   → BUY
# prob < SELL_PROB → model says price will go DOWN → SELL
# prob in between  → uncertain                     → HOLD
BUY_PROB      = 0.40     # ✨ SUPER AGGRESSIVE ✨ (was 0.50)
SELL_PROB     = 0.40     # ✨ SUPER AGGRESSIVE ✨ (was 0.40)
POS_PCT       = 0.50     # Fraction of net worth to deploy
MAX_POS_PCT   = 0.60     # Server-enforced max position
STOP_LOSS     = 0.02     # Fixed stop-loss fallback (2%)
TP_TIER1_PCT  = 0.015    # Take-profit tier 1 (1.5%)
TP_TIER2_TRAIL= 0.005    # Trailing stop for tier 2 (0.5%)
BREAKEVEN_PCT = 0.010    # Move SL to breakeven at 1.0% profit
FEE_PCT       = 0.001    # 0.1% transaction fee
COOLDOWN_TICKS= 3        # ~30 seconds cooldown after trade
DRAWDOWN_KILL = 0.03     # Kill switch at 3% drawdown
RSI_BUY_MAX   = 65       # Don't buy if RSI > 65 (was 60, slightly relaxed)
RR_MIN        = 1.2      # Minimum risk/reward ratio (was 1.5, relaxed to trigger more buys)
ATR_SPIKE_MULT= 3.0      # Sit out if candle > 3x ATR

MODEL_PATH    = "models/xgb_model.pkl"

# ═══════════════════════════════════════════════════════════════════════════
# API HELPERS (exact hackathon format)
# ═══════════════════════════════════════════════════════════════════════════
def get_price():
    r = requests.get(f"{API_URL}/api/price", headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()

def get_portfolio():
    r = requests.get(f"{API_URL}/api/portfolio", headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()

def get_history():
    """Fetch recent historical ticks to pre-fill the buffer and avoid the 30-tick wait."""
    r = requests.get(f"{API_URL}/api/history", headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()

def buy(qty):
    r = requests.post(f"{API_URL}/api/buy", json={"quantity": qty}, headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()

def sell(qty):
    r = requests.post(f"{API_URL}/api/sell", json={"quantity": qty}, headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════
def load_model(path):
    """Load pre-trained ensemble model and feature names."""
    if not os.path.exists(path):
        print(f"❌ Model not found at {path}. Run train.py first.")
        sys.exit(1)
    data = joblib.load(path)
    if isinstance(data, dict):
        return data['model'], data.get('feature_names', [])
    return data, []

# ═══════════════════════════════════════════════════════════════════════════
# INLINE FEATURE ENGINE
# Replicates processor.py exactly — self-contained, no imports from src/
# ═══════════════════════════════════════════════════════════════════════════
def compute_features(ohlcv_buffer):
    """
    Compute all 58 features from an OHLCV buffer (list of dicts).
    Returns a DataFrame with a single row (latest tick's features).
    Keys expected: 'close' (required), 'open','high','low','volume' (optional).
    """
    df = pd.DataFrame(ohlcv_buffer)
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    close = 'close'
    has_ohlc = all(c in df.columns for c in ['open', 'high', 'low', 'close'])
    has_vol = 'volume' in df.columns

    # ── Rolling Window Features ──
    for w in [3, 5, 10, 30]:
        roll = df[close].rolling(window=w)
        df[f'sma_{w}'] = roll.mean()
        df[f'sd_{w}'] = roll.std()
        df[f'z_score_{w}'] = (df[close] - df[f'sma_{w}']) / (df[f'sd_{w}'] + 1e-8)
        df[f'bb_upper_{w}'] = df[f'sma_{w}'] + (2 * df[f'sd_{w}'])
        df[f'bb_lower_{w}'] = df[f'sma_{w}'] - (2 * df[f'sd_{w}'])
        df[f'bb_width_{w}'] = (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}']) / (df[f'sma_{w}'] + 1e-8)
        df[f'bb_pos_{w}'] = (df[close] - df[f'bb_lower_{w}']) / (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}'] + 1e-8)

    # ── RSI ──
    delta = df[close].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # ── MACD ──
    ema_12 = df[close].ewm(span=12, adjust=False).mean()
    ema_26 = df[close].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ── ATR ──
    if has_ohlc:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        df['atr_pct'] = df['atr_14'] / (df['close'] + 1e-8)
    else:
        # Approximate ATR from close-only data using returns
        df['atr_14'] = df[close].diff().abs().rolling(window=14).mean()
        df['atr_pct'] = df['atr_14'] / (df[close] + 1e-8)

    # ── Candle Patterns ──
    if has_ohlc:
        df['candle_body'] = df['close'] - df['open']
        df['candle_range'] = df['high'] - df['low']
        df['body_to_range'] = df['candle_body'] / (df['candle_range'] + 1e-8)
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    else:
        df['candle_body'] = 0.0
        df['candle_range'] = 0.0
        df['body_to_range'] = 0.0
        df['upper_shadow'] = 0.0
        df['lower_shadow'] = 0.0

    # ── Stochastic Oscillator ──
    stoch_period = 14
    df['stoch_k'] = ((df[close] - df[close].rolling(stoch_period).min()) /
                     (df[close].rolling(stoch_period).max() -
                      df[close].rolling(stoch_period).min() + 1e-8)) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ── ROC ──
    df['roc_10'] = df[close].pct_change(periods=10) * 100

    # ── Volume Features ──
    if has_vol:
        df['vol_sma_10'] = df['volume'].rolling(window=10).mean()
        df['rvol_10'] = df['volume'] / (df['vol_sma_10'] + 1e-8)
        if has_ohlc:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            cum_tp_vol = (typical_price * df['volume']).rolling(window=20).sum()
            cum_vol = df['volume'].rolling(window=20).sum()
            df['vwap_20'] = cum_tp_vol / (cum_vol + 1e-8)
            df['vwap_dev'] = (df['close'] - df['vwap_20']) / (df['vwap_20'] + 1e-8)
        else:
            df['vwap_20'] = df[close].rolling(window=20).mean()
            df['vwap_dev'] = 0.0
    else:
        df['vol_sma_10'] = 0.0
        df['rvol_10'] = 1.0
        df['vwap_20'] = df[close].rolling(window=20).mean()
        df['vwap_dev'] = 0.0

    # ── Momentum & Lags ──
    df['return_1m'] = df[close].pct_change()
    df['return_5m'] = df[close].pct_change(periods=5)
    df['lag_return_1'] = df['return_1m'].shift(1)
    df['lag_return_2'] = df['return_1m'].shift(2)
    df['lag_return_3'] = df['return_1m'].shift(3)
    df['volatility_5'] = df['return_1m'].rolling(window=5).std()
    df['return_accel'] = df['return_1m'].diff()

    # ── Cleanup ──
    df = df.replace([np.inf, -np.inf], 0)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.dropna(subset=numeric_cols)

    if len(df) == 0:
        return None

    df[numeric_cols] = df[numeric_cols].clip(lower=-1e9, upper=1e9)
    return df.iloc[-1:]


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS (for safety layers)
# ═══════════════════════════════════════════════════════════════════════════
def compute_adx(closes, highs=None, lows=None, period=14):
    """Compute ADX from price data. Uses close-only approximation if no H/L."""
    n = len(closes)
    if n < period * 2:
        return 20.0  # Neutral default

    if highs is not None and lows is not None:
        h, l, c = np.array(highs), np.array(lows), np.array(closes)
        plus_dm = np.maximum(np.diff(h), 0)
        minus_dm = np.maximum(-np.diff(l), 0)
        # Zero out when the other is larger
        mask = plus_dm > minus_dm
        minus_dm[mask & (plus_dm > minus_dm)] = 0
        plus_dm[~mask] = 0
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    else:
        # Close-only approximation: use absolute returns as directional proxy
        c = np.array(closes)
        diff = np.diff(c)
        plus_dm = np.where(diff > 0, np.abs(diff), 0)
        minus_dm = np.where(diff < 0, np.abs(diff), 0)
        tr = np.abs(diff)

    tr[tr == 0] = 1e-8

    # Smoothed averages
    def smooth(arr, p):
        s = np.zeros(len(arr))
        s[p-1] = np.mean(arr[:p])
        for i in range(p, len(arr)):
            s[i] = s[i-1] - s[i-1]/p + arr[i]
        return s

    atr_s = smooth(tr, period)
    plus_di = 100 * smooth(plus_dm, period) / (atr_s + 1e-8)
    minus_di = 100 * smooth(minus_dm, period) / (atr_s + 1e-8)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    adx = smooth(dx[period:], period)

    return float(adx[-1]) if len(adx) > 0 and adx[-1] > 0 else 20.0


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1: PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════
def pre_flight_check(closes, highs, lows, volumes, tick_count, total_ticks, has_ohlc, has_vol):
    """
    Returns (pass_bool, regime, reason).
    regime: 'trend' or 'reversion'
    """
    if len(closes) < 31:
        return False, 'buffering', 'Buffering data'

    # ── Session Lock: No new entries in last 15 minutes (90 ticks) ──
    if total_ticks > 0 and tick_count > (total_ticks - 90):
        return False, 'locked', 'Session end lock — no new entries'

    # ── ADX Regime Detection ──
    adx = compute_adx(closes, highs if has_ohlc else None, lows if has_ohlc else None)
    regime = 'trend' if adx > 25 else 'reversion'

    # ── Trend Alignment: Price > 20-EMA for buys ──
    ema_20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
    trend_ok = closes[-1] >= ema_20  # Will be checked specifically for BUY signals

    # ── RVOL Check ──
    rvol_ok = True
    if has_vol and len(volumes) >= 15:
        avg_vol = np.mean(volumes[-15:])
        rvol_ok = volumes[-1] > avg_vol * 1.2 if avg_vol > 0 else True

    return True, regime, f'regime={regime} adx={adx:.1f} trend_ok={trend_ok} rvol_ok={rvol_ok}'


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2: ML SIGNAL
# ═══════════════════════════════════════════════════════════════════════════
def get_ml_signal(features_row, model, feature_names):
    """Run ensemble prediction. Returns (action, probability)."""
    if features_row is None:
        return 'hold', 0.5

    # Ensure feature order matches training
    if feature_names:
        # Only use features that exist in both
        available = [f for f in feature_names if f in features_row.columns]
        missing = [f for f in feature_names if f not in features_row.columns]
        if missing:
            for m in missing:
                features_row[m] = 0.0
        features_row = features_row[feature_names]

    try:
        prob = model.predict_proba(features_row)[0][1]  # P(up)
    except Exception as e:
        print(f"  ⚠️ Model error: {e}")
        return 'hold', 0.5

    if prob > BUY_PROB:
        return 'buy', prob
    elif prob < SELL_PROB:
        return 'sell', prob
    return 'hold', prob


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3: ENTRY CONFLUENCE
# ═══════════════════════════════════════════════════════════════════════════
def entry_confluence(action, closes, rsi, atr, price, has_ohlc, candle_range=0):
    """
    Final confirmation before entry.
    Returns (allow_bool, reason).
    """
    if action != 'buy':
        return True, 'sell/hold — no confluence needed'

    # ── RSI Filter: Don't buy overbought ──
    if rsi > RSI_BUY_MAX:
        return False, f'RSI too high ({rsi:.1f} > {RSI_BUY_MAX})'

    # ── Volatility Spike Guard ──
    if has_ohlc and atr > 0 and candle_range > atr * ATR_SPIKE_MULT:
        return False, f'Volatility spike (candle={candle_range:.4f} > {ATR_SPIKE_MULT}x ATR={atr:.4f})'

    # ── Risk/Reward Check ──
    if atr > 0:
        stop_dist = 1.5 * atr
        target_dist = price * TP_TIER1_PCT
        rr = target_dist / (stop_dist + 1e-8)
        if rr < RR_MIN:
            return False, f'R:R too low ({rr:.2f} < {RR_MIN})'

    # ── Trend Alignment ──
    if len(closes) >= 20:
        ema_20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
        if price < ema_20:
            return False, f'Price below 20-EMA ({price:.4f} < {ema_20:.4f})'

    return True, 'All confluence checks passed'


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 4: POSITION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
class PositionManager:
    def __init__(self):
        self.entry_price = None
        self.peak_price = None
        self.tp1_hit = False          # Has tier1 take-profit been triggered?
        self.sl_price = None          # Current stop-loss level
        self.original_shares = 0     # Shares at entry

    def on_entry(self, price, shares, atr):
        self.entry_price = price
        self.peak_price = price
        self.tp1_hit = False
        self.original_shares = shares
        # ATR-based stop loss: increased from 1.5x to 5.0x to avoid microscopic 10-second noise
        self.sl_price = price - (5.0 * atr) if atr > 0 else price * (1 - STOP_LOSS)

    def check(self, price, shares):
        """
        Returns (action, qty, reason) based on position management rules.
        """
        if self.entry_price is None or shares == 0:
            return 'hold', 0, 'No position'

        # Track peak price for trailing stop
        if price > self.peak_price:
            self.peak_price = price

        pnl_pct = (price - self.entry_price) / self.entry_price

        # ── Break-Even: Move SL to entry + fee once +1.0% ──
        if pnl_pct >= BREAKEVEN_PCT and not self.tp1_hit:
            new_sl = self.entry_price * (1 + FEE_PCT)
            if new_sl > self.sl_price:
                self.sl_price = new_sl

        # ── TP Tier 1: Sell 50% at +1.5% ──
        if pnl_pct >= TP_TIER1_PCT and not self.tp1_hit:
            self.tp1_hit = True
            sell_qty = max(1, shares // 2)
            return 'sell', sell_qty, f'TP1: +{pnl_pct*100:.2f}% — selling 50%'

        # ── TP Tier 2: Trailing stop on remaining ──
        if self.tp1_hit:
            trail_sl = self.peak_price * (1 - TP_TIER2_TRAIL)
            if price <= trail_sl:
                return 'sell', shares, f'Trailing stop hit @ {price:.4f} (peak={self.peak_price:.4f})'

        # ── Stop-Loss ──
        if price <= self.sl_price:
            return 'sell', shares, f'Stop-loss hit @ {price:.4f} (SL={self.sl_price:.4f})'

        return 'hold', 0, f'Holding | P&L: {pnl_pct*100:+.2f}% | SL: {self.sl_price:.4f}'

    def on_exit(self, partial=False):
        if not partial:
            self.entry_price = None
            self.peak_price = None
            self.tp1_hit = False
            self.sl_price = None
            self.original_shares = 0


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 5: SYSTEM STABILITY
# ═══════════════════════════════════════════════════════════════════════════
class SystemGuard:
    def __init__(self, start_worth=100_000):
        self.start_worth = start_worth
        self.killed = False
        self.cooldown_until = 0       # Tick number when cooldown expires
        self.nw_history = []          # Net worth history for Sharpe

    def check(self, net_worth, tick_num):
        """Returns (allow_trading, reason)."""
        self.nw_history.append(net_worth)

        # ── Drawdown Kill Switch ──
        drawdown = (self.start_worth - net_worth) / self.start_worth
        if drawdown >= DRAWDOWN_KILL:
            self.killed = True
            return False, f'KILL SWITCH: Drawdown {drawdown*100:.2f}% >= {DRAWDOWN_KILL*100}%'

        if self.killed:
            return False, 'Trading killed — drawdown limit hit'

        # ── Cooldown ──
        if tick_num < self.cooldown_until:
            return False, f'Cooldown: {self.cooldown_until - tick_num} ticks remaining'

        return True, 'OK'

    def start_cooldown(self, tick_num):
        self.cooldown_until = tick_num + COOLDOWN_TICKS

    def get_sharpe(self):
        if len(self.nw_history) < 10:
            return 0.0
        returns = np.diff(self.nw_history) / (np.array(self.nw_history[:-1]) + 1e-8)
        std = np.std(returns)
        if std < 1e-10:
            return 0.0
        # Annualize: ~1080 ticks in 3 hours
        return float(np.clip(np.mean(returns) / std * np.sqrt(1080), -999, 999))


# ═══════════════════════════════════════════════════════════════════════════
# DECIDE: ORCHESTRATE ALL 5 LAYERS
# ═══════════════════════════════════════════════════════════════════════════
def decide(ohlcv_buffer, portfolio, price, model, feature_names, pos_mgr, sys_guard,
           tick_num, total_ticks, has_ohlc, has_vol):
    """
    Master decision function. Runs all 5 layers in sequence.
    Returns (action, qty, reason).
    """
    shares = portfolio.get('shares', 0)
    cash = portfolio.get('cash', 0)
    net_worth = portfolio.get('net_worth', cash + shares * price)

    closes = [t.get('close', t.get('Close', 0)) for t in ohlcv_buffer]
    highs = [t.get('high', t.get('High', 0)) for t in ohlcv_buffer] if has_ohlc else None
    lows = [t.get('low', t.get('Low', 0)) for t in ohlcv_buffer] if has_ohlc else None
    volumes = [t.get('volume', t.get('Volume', 0)) for t in ohlcv_buffer] if has_vol else None

    # ── LAYER 4 FIRST: Check existing position (stop-loss, take-profit) ──
    if shares > 0:
        pos_action, pos_qty, pos_reason = pos_mgr.check(price, shares)
        if pos_action == 'sell':
            return 'sell', pos_qty, f'[L4-POSITION] {pos_reason}'

    # ── LAYER 5: System stability ──
    sys_ok, sys_reason = sys_guard.check(net_worth, tick_num)
    if not sys_ok:
        return 'hold', 0, f'[L5-SYSTEM] {sys_reason}'

    # ── LAYER 1: Pre-flight ──
    pf_ok, regime, pf_reason = pre_flight_check(
        closes, highs, lows, volumes, tick_num, total_ticks, has_ohlc, has_vol
    )
    if not pf_ok:
        return 'hold', 0, f'[L1-PREFLIGHT] {pf_reason}'

    # ── LAYER 2: ML Signal ──
    features_row = compute_features(ohlcv_buffer)
    action, prob = get_ml_signal(features_row, model, feature_names)

    if action == 'hold':
        return 'hold', 0, f'[L2-ML] Hold (prob={prob:.3f})'

    # ── LAYER 3: Entry Confluence (for BUY only) ──
    if action == 'buy':
        rsi = float(features_row['rsi_14'].iloc[0]) if features_row is not None and 'rsi_14' in features_row.columns else 50.0
        atr = float(features_row['atr_14'].iloc[0]) if features_row is not None and 'atr_14' in features_row.columns else 0.0
        candle_range = float(features_row['candle_range'].iloc[0]) if features_row is not None and 'candle_range' in features_row.columns else 0.0

        conf_ok, conf_reason = entry_confluence(action, closes, rsi, atr, price, has_ohlc, candle_range)
        if not conf_ok:
            return 'hold', 0, f'[L3-CONFLUENCE] {conf_reason}'

        # ── Position sizing with 60% cap ──
        max_shares = int((net_worth * MAX_POS_PCT) / price)
        can_buy = max(0, max_shares - shares)
        target_shares = int((net_worth * POS_PCT) / price)
        want_buy = max(0, target_shares - shares)
        affordable = int(cash * 0.99 / (price * (1 + FEE_PCT))) if price > 0 else 0
        qty = min(want_buy, can_buy, affordable)

        if qty <= 0:
            return 'hold', 0, '[L4-SIZE] Already at max position or no cash'

        return 'buy', qty, f'[L2-ML] BUY (prob={prob:.3f}, qty={qty}, regime={regime})'

    elif action == 'sell':
        if shares <= 0:
            return 'hold', 0, '[L2-ML] Sell signal but no shares'
        else:
            # ✨ AGGRESSIVE MODE ✨
            # We bought aggressively, so we must HOLD aggressively. 
            # We ignore the ML model's raw sell signals while in a position,
            # trusting our 5.0x ATR Stop-Loss and Take-Profit tiers to handle the exit.
            return 'hold', 0, f'[L2-ML] Ignoring ML Sell (prob={prob:.3f}) to let position run'

    return 'hold', 0, f'[L2-ML] Hold (prob={prob:.3f})'


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  🚀 AlgoTrading Bot — 5-Layer Safety Framework")
    print("=" * 60)
    print(f"  API:   {API_URL}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Key:   {'✅ loaded' if API_KEY and API_KEY != 'YOUR_KEY_HERE' else '❌ MISSING'}")
    print("=" * 60)

    # Load ML model
    model, feature_names = load_model(MODEL_PATH)
    print(f"✅ Model loaded ({len(feature_names)} features)")

    # State
    ohlcv_buffer = []
    pos_mgr = PositionManager()
    sys_guard = SystemGuard()
    tick_num = 0
    has_ohlc = None  # Auto-detect on first tick
    has_vol = None
    total_ticks = 1080  # 3 hours × 6 ticks/min (estimate, will update from API)

    # ── Pre-fill Buffer using History API ──
    print("\n⏳ Pre-filling buffer from /api/history to skip the 5-minute wait...")
    try:
        hist_data = get_history()
        if isinstance(hist_data, list):
            for t in hist_data[-100:]:  # Take last 100 max
                t_norm = {k.lower(): v for k, v in t.items()}
                ohlcv_buffer.append(t_norm)
            print(f"✅ Loaded {len(ohlcv_buffer)} historical ticks. Ready to trade instantly!")
        else:
            print("⚠️ /api/history did not return a list. Falling back to live buffering.")
    except Exception as e:
        print(f"⚠️ Could not fetch history ({e}). Will buffer live data.")

    print("\n🟢 Bot is LIVE. Ctrl+C to stop.\n")

    while True:
        try:
            # ── Fetch data ──
            tick = get_price()
            port = get_portfolio()
            price = tick.get('close', tick.get('Close', 0))

            # ── Auto-detect available fields on first tick ──
            if has_ohlc is None:
                has_ohlc = all(k in tick for k in ['open', 'high', 'low']) or all(k in tick for k in ['Open', 'High', 'Low'])
                has_vol = 'volume' in tick or 'Volume' in tick
                print(f"  📡 API fields: OHLC={'✅' if has_ohlc else '❌ (close-only)'}  Volume={'✅' if has_vol else '❌'}")

            # ── Check market phase ──
            phase = tick.get('phase', 'live')
            if phase == 'closed':
                print("\n🏁 Market closed.")
                break

            # ── Normalize and buffer ──
            tick_norm = {k.lower(): v for k, v in tick.items()}
            ohlcv_buffer.append(tick_norm)
            if len(ohlcv_buffer) > 200:
                ohlcv_buffer.pop(0)

            # ── Decide ──
            action, qty, reason = decide(
                ohlcv_buffer, port, price, model, feature_names,
                pos_mgr, sys_guard, tick_num, total_ticks, 
                has_ohlc or False, has_vol or False
            )

            # ── Execute ──
            if action == 'buy' and qty > 0:
                result = buy(qty)
                atr_val = 0
                if len(ohlcv_buffer) >= 14:
                    features = compute_features(ohlcv_buffer)
                    if features is not None and 'atr_14' in features.columns:
                        atr_val = float(features['atr_14'].iloc[0])
                pos_mgr.on_entry(price, qty + port.get('shares', 0), atr_val)
                sys_guard.start_cooldown(tick_num)
                print(f"  ⚡ BUY {qty} @ {price:.4f} | {reason}")

            elif action == 'sell' and qty > 0:
                result = sell(qty)
                partial = qty < port.get('shares', 0)
                pos_mgr.on_exit(partial=partial)
                sys_guard.start_cooldown(tick_num)
                print(f"  🔻 SELL {qty} @ {price:.4f} | {reason}")

            else:
                # ── Heartbeat every tick ──
                sharpe = sys_guard.get_sharpe()
                print(f"  💓 tick={tick_num} | {price:.4f} | nw=${port.get('net_worth',0):,.0f} | "
                      f"pnl={port.get('pnl_pct',0):+.2f}% | sharpe={sharpe:.2f} | {reason}")

            tick_num += 1
            time.sleep(10)

        except KeyboardInterrupt:
            print("\n🛑 Bot stopped.")
            break
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  ⚠️ Rate limited (429). Waiting 10s...")
            else:
                print(f"  ⚠️ HTTP Error: {e}")
            time.sleep(10)
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            time.sleep(10)

    # ── Final Stats ──
    print("\n" + "=" * 60)
    print("  📊 FINAL STATS")
    print("=" * 60)
    try:
        final_port = get_portfolio()
        print(f"  Net Worth:  ${final_port.get('net_worth', 0):,.2f}")
        print(f"  P&L:        {final_port.get('pnl_pct', 0):+.2f}%")
        print(f"  Cash:       ${final_port.get('cash', 0):,.2f}")
        print(f"  Shares:     {final_port.get('shares', 0)}")
    except:
        pass
    print(f"  Ticks:      {tick_num}")
    print(f"  Sharpe:     {sys_guard.get_sharpe():.4f}")
    print("=" * 60)
