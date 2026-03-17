"""
agent.py — Asset Alpha Trading Bot  (Kalman + Online ML Edition)

New techniques vs previous version:
  1. Kalman Filter  — smooths price noise, estimates velocity & next-tick prediction
  2. Online SGDClassifier — learns from every live tick, adapts to current market
  3. Volatility-adjusted sizing — bets more when market is calm, less when wild
  4. Fee-aware minimum threshold — never trades unless expected profit > fee cost

API:
  GET  /api/price      → {close, phase, tick_number, volume}
  GET  /api/portfolio  → {cash, shares, net_worth, pnl_pct}
  POST /api/buy        → {quantity: int}
  POST /api/sell       → {quantity: int}
  Header: X-API-Key
"""
import os, time, requests, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────
API_URL  = os.getenv("API_URL",      "https://algotrading.sanyamchhabra.in")
API_KEY  = os.getenv("TEAM_API_KEY", "ak_ff892bf94ef82a3708cee7ba4df4f181")
HEADERS  = {"X-API-Key": API_KEY}

MODEL_PATH  = "models/xgb_model.pkl"
TICK_SLEEP  = 10
EST_FEE_PCT = 0.0007
MIN_PROFIT  = EST_FEE_PCT * 2.5
BASE_PCT    = 0.35

# ── API ────────────────────────────────────────────────────────────────
def _cast(v): return v[0] if isinstance(v, (list, tuple)) else v

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
        'shares':    int(_cast(raw.get('shares',      0))),
        'net_worth': float(_cast(raw.get('net_worth', 100000))),
        'pnl_pct':   float(_cast(raw.get('pnl_pct',  0))),
    }

def buy(qty: int):
    if qty <= 0: return None
    r = requests.post(f"{API_URL}/api/buy", json={"quantity": qty},
                      headers=HEADERS, timeout=5)
    if r.status_code == 400:
        print(f"  ⚠️  BUY rejected (400): {r.text[:100]}")
        return None
    r.raise_for_status()
    return r.json()

def sell(qty: int):
    if qty <= 0: return None
    r = requests.post(f"{API_URL}/api/sell", json={"quantity": qty},
                      headers=HEADERS, timeout=5)
    if r.status_code == 400:
        print(f"  ⚠️  SELL rejected (400): {r.text[:100]}")
        return None
    r.raise_for_status()
    return r.json()

# ══════════════════════════════════════════════════════════════════════
# KALMAN FILTER
# ══════════════════════════════════════════════════════════════════════
class KalmanFilter:
    def __init__(self, process_noise=0.05, obs_noise=0.5):
        self.x = None
        self.P = np.eye(2) * 1000.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.diag([process_noise, process_noise * 0.1])
        self.R = np.array([[obs_noise]])

    def update(self, price):
        if self.x is None:
            self.x = np.array([price, 0.0])
            return price, 0.0, price
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = np.array([price]) - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + (K @ y).ravel()
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        return self.x[0], self.x[1], self.x[0] + self.x[1]

# ══════════════════════════════════════════════════════════════════════
# ONLINE MODEL
# ══════════════════════════════════════════════════════════════════════
class OnlineModel:
    def __init__(self, pretrained_path=MODEL_PATH):
        self.clf    = SGDClassifier(loss='log_loss', learning_rate='adaptive',
                                    eta0=0.01, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False
        self.n_seen = 0
        self.pretrained  = None
        self.pt_features = None
        if os.path.exists(pretrained_path):
            try:
                data = joblib.load(pretrained_path)
                if isinstance(data, dict):
                    self.pretrained  = data['model']
                    self.pt_features = data.get('feature_names')
                else:
                    self.pretrained = data
                print(f"✅ Pretrained ensemble loaded from {pretrained_path}")
            except Exception as e:
                print(f"⚠️  Could not load pretrained model: {e}")

    def _feats(self, closes, volumes, kl, kv, kp):
        p   = np.array(closes, dtype=float)
        vol = np.array(volumes, dtype=float)
        n   = len(p)
        f   = []
        f.append(kv)
        f.append(kp - p[-1])
        f.append((p[-1] - kl) / (abs(kl) + 1e-8))
        f.append((p[-1]-p[-2])/p[-2] if n>=2 else 0.0)
        f.append((p[-1]-p[-3])/p[-3] if n>=3 else 0.0)
        f.append((p[-1]-p[-5])/p[-5] if n>=5 else 0.0)
        if n >= 6:
            rets = np.diff(p[-6:]) / p[-6:-1]
            f.append(np.std(rets))
            f.append(rets[-1]-rets[-2])
        else:
            f += [0.0, 0.0]
        f.append((p[-1]-p[-10:].mean())/(p[-10:].std()+1e-8) if n>=10 else 0.0)
        f.append((p[-1]-p[-20:].mean())/(p[-20:].std()+1e-8) if n>=20 else 0.0)
        f.append(vol[-1]/(vol[-10:].mean()+1e-8) if n>=10 else 1.0)
        return np.array(f, dtype=float)

    def learn(self, closes, volumes, kl, kv, kp):
        if len(closes) < 7: return
        label = 1 if closes[-1] > closes[-2] else 0
        feats = self._feats(closes[:-1], volumes[:-1], kl, kv, kp)
        feats = np.nan_to_num(feats).reshape(1, -1)
        self.scaler.partial_fit(feats)
        feats_s = self.scaler.transform(feats)
        self.clf.partial_fit(feats_s, [label], classes=[0,1])
        self.fitted = True
        self.n_seen += 1

    def predict(self, closes, volumes, kl, kv, kp):
        if len(closes) < 7: return None
        feats = np.nan_to_num(self._feats(closes, volumes, kl, kv, kp)).reshape(1,-1)
        online_prob = None
        if self.fitted and self.n_seen >= 15:
            feats_s = self.scaler.transform(feats)
            online_prob = float(self.clf.predict_proba(feats_s)[0][1])
        pt_prob = None
        if self.pretrained is not None and len(closes) >= 35:
            try:
                p = np.array(closes, dtype=float)
                row = {}
                for w in [3,5,10,30]:
                    if len(p)>=w:
                        sma=p[-w:].mean(); std=p[-w:].std()+1e-8
                        row[f'sma_{w}']=sma; row[f'sd_{w}']=std
                        row[f'z_score_{w}']=(p[-1]-sma)/std
                        bu=sma+2*std; bl=sma-2*std
                        row[f'bb_upper_{w}']=bu; row[f'bb_lower_{w}']=bl
                        row[f'bb_width_{w}']=(bu-bl)/(sma+1e-8)
                        row[f'bb_pos_{w}']=(p[-1]-bl)/(bu-bl+1e-8)
                if len(p)>=15:
                    d=np.diff(p[-15:]); g=np.mean(np.where(d>0,d,0)); l=np.mean(np.where(d<0,-d,0))
                    row['rsi_14']=100-(100/(1+g/(l+1e-8)))
                if len(p)>=26:
                    s=pd.Series(p)
                    macd=s.ewm(span=12,adjust=False).mean().iloc[-1]-s.ewm(span=26,adjust=False).mean().iloc[-1]
                    sig=s.ewm(span=9,adjust=False).mean().iloc[-1]
                    row['macd']=macd; row['macd_signal']=sig; row['macd_hist']=macd-sig
                if len(p)>=14:
                    lo=p[-14:].min(); hi=p[-14:].max(); k=(p[-1]-lo)/(hi-lo+1e-8)*100
                    row['stoch_k']=k; row['stoch_d']=k
                if len(p)>=11: row['roc_10']=(p[-1]-p[-11])/(p[-11]+1e-8)*100
                v=np.array(volumes,dtype=float)
                if len(v)>=10: row['vol_sma_10']=v[-10:].mean(); row['rvol_10']=v[-1]/(v[-10:].mean()+1e-8)
                for lag,attr in [(2,'return_1m'),(6,'return_5m')]:
                    if len(p)>=lag: row[attr]=(p[-1]-p[-lag])/(p[-lag]+1e-8)
                for lag,attr in [(3,'lag_return_1'),(4,'lag_return_2'),(5,'lag_return_3')]:
                    if len(p)>=lag: row[attr]=(p[-lag+1]-p[-lag])/(p[-lag]+1e-8)
                if len(p)>=6:
                    rets=np.diff(p[-6:])/(p[-6:-1]+1e-8)
                    row['volatility_5']=np.std(rets)
                    row['return_accel']=rets[-1]-rets[-2] if len(rets)>=2 else 0
                cols = self.pt_features if self.pt_features else list(row.keys())
                X = pd.DataFrame([{k: row.get(k,0.0) for k in cols}])[cols]
                X = X.replace([np.inf,-np.inf],0).fillna(0)
                raw = self.pretrained.predict_proba(X)
                pt_prob = float(np.array(raw).ravel()[1]) if isinstance(raw,(list,tuple)) else float(raw[0][1])
            except Exception:
                pt_prob = None
        if online_prob is not None and pt_prob is not None:
            w = min(self.n_seen/50.0, 0.7)
            return w*online_prob + (1-w)*pt_prob
        return online_prob if online_prob is not None else pt_prob

# ── Position sizing ────────────────────────────────────────────────────
def position_size(cash, price, vol5, base=BASE_PCT):
    vf  = min(0.0005/(vol5+1e-8), 2.0)
    pct = max(0.15, min(0.40, base*vf))
    return max(1, int(cash*pct/price)), pct

# ══════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════
class SignalEngine:
    def __init__(self):
        self.kf    = KalmanFilter()
        self.model = OnlineModel()
        self.kf_level = None
        self.kf_vel   = 0.0
        self.kf_pred  = None
        self.prev_kl  = None
        self.prev_kv  = 0.0
        self.prev_kp  = None

    def update_kalman(self, price):
        self.prev_kl, self.prev_kv, self.prev_kp = self.kf_level, self.kf_vel, self.kf_pred
        self.kf_level, self.kf_vel, self.kf_pred = self.kf.update(price)

    def learn(self, closes, volumes):
        if self.prev_kl is not None:
            self.model.learn(closes, volumes, self.prev_kl, self.prev_kv, self.prev_kp)

    def decide(self, closes, volumes, portfolio, price):
        n       = len(closes)
        holding = portfolio['shares'] > 0
        cash    = portfolio['cash']

        if n < 7 or self.kf_level is None:
            return 'warmup', 0, {}

        kl, kv, kp = self.kf_level, self.kf_vel, self.kf_pred
        dev        = price - kl
        prob       = self.model.predict(closes, volumes, kl, kv, kp)
        p          = np.array(closes, dtype=float)
        vol5       = np.std(np.diff(p[-6:])/p[-6:-1]) if n>=6 else 0.001
        atr        = np.mean(np.abs(np.diff(p[-15:]))) if n>=15 else (price * 0.001)
        up_count   = sum(1 for i in range(-5,-1) if p[i+1]>p[i]) if n>=6 else 2
        regime     = 'up' if up_count>=2 else ('down' if up_count<=1 else 'choppy')
        
        # 5-tick future prediction
        kp_5       = kl + kv * 5
        pred_move_5 = (kp_5 - price) / price
        
        ml_bull    = prob is not None and prob >= 0.50
        ml_bear    = prob is not None and prob <= 0.49
        last_exit  = portfolio.get('last_exit_price', 0)
        chasing    = last_exit > 0 and price >= last_exit * 1.001

        action, qty, reason = 'hold', 0, ''

        if not holding and not chasing:
            buy_sig = False
            # Less defensive: allow buying if predicted 5-tick move is positive or slightly negative
            if regime == 'up' and kv > -0.005 and pred_move_5 > -0.005:
                buy_sig = True; reason = f'trend+kv={kv:+.4f}'
            elif dev < -0.010*vol5*1000 and pred_move_5 > -0.01:
                if ml_bull or prob is None or (regime == 'choppy'):
                    buy_sig = True; reason = f'reversion dev={dev:+.4f}'
            
            if buy_sig and pred_move_5 < MIN_PROFIT * 0.5:
                buy_sig = False; reason = 'fee_threshold_5t'
            if buy_sig:
                qty, _ = position_size(cash, price, vol5)
                action = 'buy'

        elif holding:
            if regime == 'down' and kv < -0.01:
                action='sell'; qty=portfolio['shares']; reason='downtrend'
            elif kv < -0.008 and pred_move_5 < -0.005:
                action='sell'; qty=portfolio['shares']; reason=f'kv_reversal={kv:+.4f}'
            elif ml_bear and pred_move_5 < -0.005:
                action='sell'; qty=portfolio['shares']; reason=f'ml_bear p={prob:.3f}'
            elif regime=='choppy' and dev > 0.1*vol5*1000:
                action='sell'; qty=portfolio['shares']; reason='reversion_top'

        return action, qty, {
            'kl':kl,'kv':kv,'kp':kp,'kp_5':kp_5,'dev':dev,'prob':prob,
            'vol5':vol5,'atr':atr,'regime':regime,'reason':reason,
            'n_online':self.model.n_seen
        }

# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def run():
    print("🚀 Asset Alpha — Kalman + Online ML Edition")
    print(f"   Server : {API_URL}")
    try:
        tick = get_price(); port = get_portfolio()
        print(f"✅ Connected | price={tick['close']:.4f} | phase={tick['phase']} | tick={tick['tick_number']}")
        print(f"   cash=${port['cash']:,.2f} | shares={port['shares']} | nw=${port['net_worth']:,.2f} | pnl={port['pnl_pct']:+.2f}%")
    except Exception as e:
        print(f"❌ Connection failed: {e}"); return

    engine          = SignalEngine()
    closes          = []
    volumes         = []
    entry_price     = None
    peak_price      = None
    last_exit_price = 0
    last_trade_tick = 0

    print("\n🟢 Bot live. Ctrl+C to stop.\n")
    sl_price = 0.0

    while True:
        try:
            tick     = get_price()
            port     = get_portfolio()
            price    = tick['close']
            tick_num = tick['tick_number']

            if tick['phase'] == 'closed':
                print("🔔 Market closed."); break

            closes.append(price)
            volumes.append(tick.get('volume', 1))
            if len(closes) > 300:
                closes.pop(0); volumes.pop(0)

            engine.update_kalman(price)
            engine.learn(closes, volumes)

            # ── Risk exits (ATR-based) ────────────────────────────────
            if port['shares'] > 0 and entry_price is not None:
                if peak_price is None or price > peak_price:
                    peak_price = price
                pnl_pct = (price - entry_price) / entry_price
                
                atr = np.mean(np.abs(np.diff(closes[-15:]))) if len(closes) >= 15 else (price * 0.001)
                
                if sl_price == 0.0:
                    sl_price = entry_price - (4.0 * atr) # Wider initial stop 4 ATR down
                
                # Breakeven condition: if up 2 ATR, move SL to entry + fees + small profit
                breakeven_dist = 2.0 * atr
                if (price - entry_price) >= breakeven_dist:
                    new_sl = entry_price * (1 + EST_FEE_PCT * 4)
                    if new_sl > sl_price:
                        sl_price = new_sl
                        
                # Trailing stop: Trail heavily by 3.5 ATR from peak once in profit
                if (peak_price - entry_price) >= breakeven_dist:
                    trail_sl = peak_price - (3.5 * atr)
                    if trail_sl > sl_price:
                        sl_price = trail_sl
                        
                hit = None
                if price <= sl_price:
                    hit = ('🛑', 'STOP-LOSS / TRAIL')
                elif pnl_pct >= 0.10: # 10% hard take profit instead of 5%
                    hit = ('🎯', 'TAKE-PROFIT')
                    
                if hit:
                    sell(port['shares']); port = get_portfolio()
                    last_exit_price = price; last_trade_tick = tick_num
                    print(f"  {'─'*70}")
                    print(f"  {hit[0]} {hit[1]} @ {price:.4f} | pnl={pnl_pct*100:+.2f}% | nw=${port.get('net_worth',0):,.2f}")
                    print(f"  {'─'*70}")
                    entry_price = None; peak_price = None; sl_price = 0.0
                    time.sleep(TICK_SLEEP); continue

            # ── Gate reset ────────────────────────────────────────────
            if last_exit_price > 0 and (tick_num - last_trade_tick) > 20:
                last_exit_price = 0

            # ── Signal ────────────────────────────────────────────────
            port['last_exit_price'] = last_exit_price
            action, qty, dbg = engine.decide(closes, volumes, port, price)

            # ── Print ─────────────────────────────────────────────────
            if len(closes) < 7:
                print(f"  ⏳ {len(closes)}/7 ticks")
            else:
                kv_s   = f"{dbg['kv']:+.4f}"
                pm_5_s = f"{((dbg['kp_5']-price)/price*100):+.4f}%"
                pr_s   = f"{dbg['prob']:.3f}" if dbg['prob'] is not None else " n/a"
                r_icon = {'up':'📈','down':'📉','choppy':'〰'}.get(dbg['regime'],'?')
                pos_s  = f"{port['shares']}sh" if port['shares']>0 else "flat"
                act_s  = f">>> {action.upper()}" if action != 'hold' else '·'
                print(f"  #{tick_num} {price:.4f} | kv={kv_s} Δ5={pm_5_s} p={pr_s} {r_icon} [{dbg['n_online']}L] | {act_s} {pos_s} | {port.get('pnl_pct',0):+.2f}%")

            # ── Execute ───────────────────────────────────────────────
            if action == 'buy' and qty > 0:
                resp = buy(qty)
                if resp:
                    entry_price = price; peak_price = price; sl_price = 0.0
                    last_trade_tick = tick_num
                    port = get_portfolio()
                    print(f"  {'─'*70}")
                    print(f"  📈 BUY  {qty} @ {price:.4f} | {dbg['reason']} | nw=${port.get('net_worth',0):,.2f}")
                    print(f"  {'─'*70}")

            elif action == 'sell' and port['shares'] > 0:
                resp = sell(qty)
                if resp:
                    tpnl = (price-entry_price)/entry_price*100 if entry_price else 0
                    last_exit_price = price; last_trade_tick = tick_num
                    port = get_portfolio()
                    icon = "✅" if tpnl >= 0 else "🔴"
                    print(f"  {'─'*70}")
                    print(f"  {icon} SELL {qty} @ {price:.4f} | trade={tpnl:+.2f}% | {dbg['reason']} | nw=${port.get('net_worth',0):,.2f}")
                    print(f"  {'─'*70}")
                    entry_price = None; peak_price = None; sl_price = 0.0

            time.sleep(TICK_SLEEP)

        except KeyboardInterrupt:
            print("\n🛑 Stopped.")
            port = get_portfolio()
            if port:
                print(f"   nw=${port.get('net_worth',0):,.2f} | P&L={port.get('pnl_pct',0):+.2f}%")
            break
        except requests.exceptions.Timeout:
            print("  ⏳ API Timeout. Retrying soon...")
            time.sleep(TICK_SLEEP * 2)
        except requests.exceptions.ConnectionError:
            print("  🔌 Connection Error. Server might be down or unreachable. Retrying...")
            time.sleep(TICK_SLEEP * 2)
        except Exception as e:
            import traceback
            print(f"  ⚠️  {e}")
            traceback.print_exc()
            time.sleep(TICK_SLEEP)

if __name__ == "__main__":
    run()