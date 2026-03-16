from flask import Flask, request, jsonify
import pandas as pd
import time
import os

app = Flask(__name__)

# Load data
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/asset_alpha_training.csv")
try:
    df = pd.read_csv(DATA_PATH)
    # Ensure columns match what the agent expects (lowercase)
    df.columns = [c.lower() for c in df.columns]
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

# Global state
class State:
    def __init__(self):
        self.tick_index = 0
        self.start_cash = 100000.0
        self.cash = 100000.0
        self.shares = 0
        self.fee = 0.001
        self.decay = 0.0002
        self.phase = "sandbox"
        
    def get_current_price(self):
        if self.tick_index < len(df):
            return df.iloc[self.tick_index]["close"]
        return 0.0
        
    def advance_tick(self):
        self.tick_index += 1
        # Apply cash decay each tick to match the starter notebook backtest logic
        self.cash *= (1 - self.decay)
        
    def reset(self):
        self.__init__()

state = State()

def verify_key():
    key = request.headers.get("X-API-Key")
    if not key:
        return False
    return True

@app.route("/api/price", methods=["GET"])
def get_price():
    if not verify_key(): return jsonify({"error": "Unauthorized"}), 401
    
    if state.tick_index >= len(df):
        state.phase = "closed"
        return jsonify({"phase": "closed", "close": 0.0})
        
    row = df.iloc[state.tick_index]
    
    # Extract OHLCV
    data = {"phase": state.phase, "tick_number": state.tick_index}
    for col in ["open", "high", "low", "close", "volume"]:
        if col in row:
            data[col] = float(row[col])
            
    # Advance tick for next query to simulate time passing (agent polls this per tick)
    state.advance_tick()
    return jsonify(data)

@app.route("/api/portfolio", methods=["GET"])
def get_portfolio():
    if not verify_key(): return jsonify({"error": "Unauthorized"}), 401
    current_price = state.get_current_price()
    net_worth = state.cash + state.shares * current_price
    pnl_pct = (net_worth / state.start_cash - 1) * 100
    
    return jsonify({
        "cash": float(state.cash),
        "shares": int(state.shares),
        "net_worth": float(net_worth),
        "pnl_pct": float(pnl_pct)
    })

@app.route("/api/buy", methods=["POST"])
def buy():
    if not verify_key(): return jsonify({"error": "Unauthorized"}), 401
    qty = request.json.get("quantity", 0)
    if qty <= 0:
        return jsonify({"status": "error", "message": "Invalid quantity"}), 400
        
    price = state.get_current_price()
    cost = qty * price * (1 + state.fee)
    
    if cost > state.cash:
        return jsonify({"status": "error", "message": "Insufficient cash"}), 400
        
    state.cash -= cost
    state.shares += qty
    
    return jsonify({"status": "success", "message": f"Bought {qty} shares"})

@app.route("/api/sell", methods=["POST"])
def sell():
    if not verify_key(): return jsonify({"error": "Unauthorized"}), 401
    qty = request.json.get("quantity", 0)
    if qty <= 0 or qty > state.shares:
        return jsonify({"status": "error", "message": "Invalid quantity"}), 400
        
    price = state.get_current_price()
    revenue = qty * price * (1 - state.fee)
    
    state.cash += revenue
    state.shares -= qty
    
    return jsonify({"status": "success", "message": f"Sold {qty} shares"})

@app.route("/api/reset", methods=["POST"])
def reset():
    state.reset()
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
