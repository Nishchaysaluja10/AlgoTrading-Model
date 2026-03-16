"""
agent.py — Asset Alpha Trading Bot
Uses the CORRECT API endpoints discovered from the starter notebook:
  GET  /api/price      → {close, phase, tick_number}
  GET  /api/portfolio  → {cash, shares, net_worth, pnl_pct}
  POST /api/buy        → {quantity: int}
  POST /api/sell       → {quantity: int}
  Header: X-API-Key (NOT Authorization: Bearer)

Run:
  python agent.py
"""
import os
import time
import requests
import numpy as np
import joblib
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────
API_URL  = os.getenv("API_URL",      "https://algotrading.sanyamchhabra.in")
API_KEY  = os.getenv("TEAM_API_KEY", "ak_ff892bf94ef82a3708cee7ba4df4f181")
HEADERS  = {"X-API-Key": API_KEY}

MODEL_PATH   = "models/xgb_model.pkl"
POS_PCT      = 0.30    # fraction of cash to deploy per trade
STOP_LOSS    = 0.02    # hard stop: exit if down 2% from entry
TAKE_PROFIT  = 0.04    # take profit at +4%
TRAIL_PCT    = 0.015   # trailing stop: lock in gains
TICK_SLEEP   = 10      # server ticks every 10 seconds

# ── API helpers (matching server spec exactly) ─────────────────────────
def _cast(raw):
    """Flatten any value the server might wrap in a list/tuple to a plain scalar."""
    if isinstance(raw, (list, tuple)):
        return raw[0]
    return raw

def get_price():
    r = requests.get(f"{API_URL}/api/price", headers=HEADERS, timeout=5)
    r.raise_for_status()
    raw = r.json()
    return {
        'close':       float(_cast(raw.get('close', raw.get('price', 0)))),
        'phase':       str(_cast(raw.get('phase', 'open'))),
        'tick_number': int(_cast(raw.get('tick_number', 0))),
        'volume':      float(_cast(raw.get('volume', 1))),
    }

def get_portfolio():
    r = requests.get(f"{API_URL}/api/portfolio", headers=HEADERS, timeout=5)
    r.raise_for_status()
    raw = r.json()
    return {
        'cash':      float(_cast(raw.get('cash',      100000))),
        'shares':    int(_cast(raw.get('shares',    0))),
        'net_worth': float(_cast(raw.get('net_worth', 100000))),
        'pnl_pct':   float(_cast(raw.get('pnl_pct',  0))),
    }

def buy(qty: int):
    if qty <= 0:
        return None
    r = requests.post(f"{API_URL}/api/buy", json={"quantity": qty}, headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()

def sell(qty: int):
    if qty <= 0:
        return None
    r = requests.post(f"{API_URL}/api/sell", json={"quantity": qty}, headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()

# ── Signal: ML model + fallback SMA crossover ──────────────────────────
class SignalEngine:
    def __init__(self, model_path=MODEL_PATH):
        self.model        = None
        self.feature_names = None
        self._load_model(model_path)

    def _load_model(self, path):
        if os.path.exists(path):
            data = joblib.load(path)
            if isinstance(data, dict):
                self.model         = data['model']
                self.feature_names = data.get('feature_names')
            else:
                self.model = data
            print(f"✅ ML model loaded from {path}")
        else:
            print(f"⚠️  No model at {path} — using SMA crossover fallback only")

    def _ml_signal(self, closes, volumes=None):
        """Build features inline (minimal version matching processor.py)."""
        if self.model is None or len(closes) < 35:
            return None

        p   = np.array(closes, dtype=float)
        vol = np.array(volumes if volumes else [1]*len(p), dtype=float)

        def safe_std(arr): return np.std(arr) if len(arr) > 1 else 1e-8

        features = {}

        # Rolling windows
        for w in [3, 5, 10, 30]:
            if len(p) >= w:
                sma  = p[-w:].mean()
                std  = safe_std(p[-w:])
                features[f'sma_{w}']      = sma
                features[f'sd_{w}']       = std
                features[f'z_score_{w}']  = (p[-1] - sma) / (std + 1e-8)
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                features[f'bb_upper_{w}'] = bb_upper
                features[f'bb_lower_{w}'] = bb_lower
                features[f'bb_width_{w}'] = (bb_upper - bb_lower) / (sma + 1e-8)
                features[f'bb_pos_{w}']   = (p[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)

        # RSI
        if len(p) >= 15:
            delta = np.diff(p[-15:])
            gain  = np.mean(np.where(delta > 0, delta, 0))
            loss  = np.mean(np.where(delta < 0, -delta, 0))
            rs    = gain / (loss + 1e-8)
            features['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        if len(p) >= 26:
            ema12 = pd.Series(p).ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = pd.Series(p).ewm(span=26, adjust=False).mean().iloc[-1]
            macd  = ema12 - ema26
            signal_line = pd.Series(p).ewm(span=9, adjust=False).mean().iloc[-1]
            features['macd']        = macd
            features['macd_signal'] = signal_line
            features['macd_hist']   = macd - signal_line

        # Stochastic
        if len(p) >= 14:
            lo  = p[-14:].min()
            hi  = p[-14:].max()
            k   = (p[-1] - lo) / (hi - lo + 1e-8) * 100
            features['stoch_k'] = k
            features['stoch_d'] = k  # simplified; close enough for inference

        # ROC
        if len(p) >= 11:
            features['roc_10'] = (p[-1] - p[-11]) / (p[-11] + 1e-8) * 100

        # Volume
        if len(vol) >= 10:
            vol_sma = vol[-10:].mean()
            features['vol_sma_10'] = vol_sma
            features['rvol_10']    = vol[-1] / (vol_sma + 1e-8)

        # Momentum & lags
        if len(p) >= 2:
            features['return_1m']     = (p[-1] - p[-2]) / (p[-2] + 1e-8)
        if len(p) >= 6:
            features['return_5m']     = (p[-1] - p[-6]) / (p[-6] + 1e-8)
        if len(p) >= 3:
            r1 = (p[-2] - p[-3]) / (p[-3] + 1e-8)
            features['lag_return_1'] = r1
        if len(p) >= 4:
            features['lag_return_2'] = (p[-3] - p[-4]) / (p[-4] + 1e-8)
        if len(p) >= 5:
            features['lag_return_3'] = (p[-4] - p[-5]) / (p[-5] + 1e-8)
        if len(p) >= 6:
            rets = np.diff(p[-6:]) / (p[-6:-1] + 1e-8)
            features['volatility_5']  = np.std(rets)
            features['return_accel']  = rets[-1] - rets[-2] if len(rets) >= 2 else 0

        # Build row in the exact column order the model was trained on
        if self.feature_names:
            row = {k: features.get(k, 0.0) for k in self.feature_names}
            X   = pd.DataFrame([row])[self.feature_names]
        else:
            X = pd.DataFrame([features])

        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        raw = self.model.predict_proba(X)
        # VotingClassifier can return either ndarray or list-of-arrays depending
        # on sklearn version — flatten defensively to a plain float
        if isinstance(raw, (list, tuple)):
            prob = float(np.array(raw).ravel()[1])
        else:
            prob = float(raw[0][1])
        return prob

    def decide(self, hist_closes, hist_volumes, portfolio, price):
        """
        Mean-reversion strategy — buys dips, sells rips.
        Works well in choppy/range-bound markets.
        Falls back to ML when we have enough data.
        """
        p   = np.array(hist_closes, dtype=float)
        n   = len(p)

        # ── Need at least 10 ticks for any signal ────────────────────
        if n < 10:
            print(f"  [warmup] {n}/10 ticks collected — waiting to trade")
            return 'hold', 0

        # ── Core indicators (all work with just 10+ ticks) ───────────

        # 1. Short z-score (10-period) — catches quick mean reversions
        mean10 = p[-10:].mean()
        std10  = p[-10:].std() + 1e-8
        z10    = (p[-1] - mean10) / std10

        # 2. Longer z-score (30-period) if we have enough data
        z30 = 0.0
        if n >= 30:
            mean30 = p[-30:].mean()
            std30  = p[-30:].std() + 1e-8
            z30    = (p[-1] - mean30) / std30

        # 3. Short-term momentum (last 3 ticks)
        momentum = (p[-1] - p[-4]) / p[-4] if n >= 4 else 0.0

        # 4. ML probability (only when we have 35+ ticks)
        prob     = self._ml_signal(hist_closes, hist_volumes)
        prob_str = f'{prob:.3f}' if prob is not None else 'n/a'

        # ── Signal logic ──────────────────────────────────────────────
        # BUY when: price is below mean (z < -1.0) AND momentum starting to recover
        # SELL when: price is above mean (z > 1.0) AND momentum fading
        # ML confirms or overrides when available

        buy_score  = 0
        sell_score = 0

        # Z-score signals (main driver)
        if z10 < -1.2:   buy_score  += 2
        elif z10 < -0.8: buy_score  += 1
        if z10 > 1.2:    sell_score += 2
        elif z10 > 0.8:  sell_score += 1

        if z30 < -1.0:   buy_score  += 1
        if z30 > 1.0:    sell_score += 1

        # Momentum confirmation — don't buy into a falling knife
        if momentum > 0:  buy_score  += 1  # price recovering → confirms buy
        if momentum < 0:  sell_score += 1  # price falling    → confirms sell

        # ML overlay
        if prob is not None:
            if prob >= 0.53:  buy_score  += 2
            elif prob <= 0.47: sell_score += 2

        print(f"  [tick={n}] z10={z10:+.2f} z30={z30:+.2f} mom={momentum*100:+.3f}% prob={prob_str} | buy={buy_score} sell={sell_score}")

        # Need score >= 3 to act (prevents noise trades)
        if buy_score >= 3 and buy_score > sell_score:
            qty = max(0, int(portfolio['cash'] * POS_PCT / price))
            return 'buy', qty
        elif sell_score >= 3 and sell_score > buy_score:
            return 'sell', portfolio['shares']

        return 'hold', 0


# ── Main loop ──────────────────────────────────────────────────────────
def run():
    print("🚀 Asset Alpha Bot starting...")
    print(f"   Server : {API_URL}")
    print(f"   Model  : {MODEL_PATH}")

    # Connection test
    try:
        import requests as _req
        _raw_price = _req.get(f"{API_URL}/api/price", headers=HEADERS, timeout=5).json()
        _raw_port  = _req.get(f"{API_URL}/api/portfolio", headers=HEADERS, timeout=5).json()
        print(f"🔍 Raw price response : {_raw_price}")
        print(f"🔍 Raw portfolio response: {_raw_port}")
        tick = get_price()
        port = get_portfolio()
        print(f"✅ Connected | price={tick['close']:.4f} | phase={tick.get('phase')} | tick={tick.get('tick_number')}")
        print(f"   cash=${port['cash']:,.0f} | shares={port['shares']} | nw=${port['net_worth']:,.0f} | pnl={port['pnl_pct']:+.2f}%")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    engine       = SignalEngine()
    hist_closes  = []
    hist_volumes = []
    entry_price  = None
    peak_price   = None

    # ── No pre-fill needed — strategy works from tick 10 onwards ──────
    # Training CSV is at price ~132, live market is ~100 — seeding from
    # CSV would corrupt z-scores. Strategy requires only 10 ticks minimum.
    print("ℹ️  No warmup needed — will start signalling after 10 live ticks (~1.5 min)")

    print("\n🟢 Bot live. Ctrl+C to stop.\n")

    while True:
        try:
            tick = get_price()
            port = get_portfolio()
            price = tick['close']
            phase = tick.get('phase', 'open')

            if phase == 'closed':
                print("🔔 Market closed. Exiting.")
                break

            hist_closes.append(price)
            hist_volumes.append(tick.get('volume', 1))

            # Keep buffer bounded
            if len(hist_closes) > 300:
                hist_closes.pop(0)
                hist_volumes.pop(0)

            # ── Extract tick number for logging ──────────────────────
            tick_num = tick['tick_number']

            # ── Risk management (checked before new signals) ──────────
            if port['shares'] > 0 and entry_price is not None:
                # Update trailing peak
                if peak_price is None or price > peak_price:
                    peak_price = price

                pnl_pct = (price - entry_price) / entry_price

                if pnl_pct < -STOP_LOSS:
                    sell(port['shares'])
                    print(f"🛑 STOP-LOSS  @ {price:.4f} | pnl={pnl_pct*100:+.2f}%")
                    entry_price = None; peak_price = None
                    time.sleep(TICK_SLEEP); continue

                elif pnl_pct >= TAKE_PROFIT:
                    sell(port['shares'])
                    print(f"✅ TAKE-PROFIT @ {price:.4f} | pnl={pnl_pct*100:+.2f}%")
                    entry_price = None; peak_price = None
                    time.sleep(TICK_SLEEP); continue

                elif price < peak_price * (1 - TRAIL_PCT):
                    sell(port['shares'])
                    print(f"📉 TRAIL-STOP  @ {price:.4f} | peak={peak_price:.4f} | pnl={pnl_pct*100:+.2f}%")
                    entry_price = None; peak_price = None
                    time.sleep(TICK_SLEEP); continue

            # ── Signal ────────────────────────────────────────────────
            action, qty = engine.decide(hist_closes, hist_volumes, port, price)

            if action == 'buy' and qty > 0 and port['shares'] == 0:
                resp = buy(qty)
                if resp:
                    entry_price = price
                    peak_price  = price
                    print(f"📈 BUY  {qty:4d} @ {price:.4f} | tick={tick_num} | cash=${port['cash']:,.0f} | nw=${port['net_worth']:,.0f}")

            elif action == 'sell' and port['shares'] > 0:
                resp = sell(port['shares'])
                if resp:
                    pnl = (price - entry_price) / entry_price * 100 if entry_price else 0
                    print(f"📉 SELL {port['shares']:4d} @ {price:.4f} | tick={tick_num} | pnl={pnl:+.2f}% | nw=${port['net_worth']:,.0f}")
                    entry_price = None; peak_price = None

            else:
                print(f"⏸  HOLD | tick={tick['tick_number']} | price={price:.4f} | shares={port['shares']} | pnl={port['pnl_pct']:+.2f}%")

            time.sleep(TICK_SLEEP)

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            port = get_portfolio()
            print(f"   Final net worth: ${port['net_worth']:,.0f} | P&L: {port['pnl_pct']:+.2f}%")
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            print(f"⚠️  Error: {e}")
            traceback.print_exc()
            time.sleep(TICK_SLEEP)


if __name__ == "__main__":
    run()