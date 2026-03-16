"""
Config — AlgoTrading Bot (Hackathon Final)
"""

# === SERVER CONNECTION ===
API_BASE_URL = "http://YOUR_SERVER_IP:PORT/api"   # ← Replace with real server URL
API_KEY = "YOUR_API_KEY"                           # ← Replace with real API key

# === DATA ===
DATA_PATH = "data/asset_alpha_training.csv"
TARGET_COL = "close" 
MODEL_SAVE_PATH = "models/xgb_model.pkl"

# === RISK MANAGEMENT ===
STARTING_CAPITAL = 100000   
RISK_PER_TRADE = 0.05       

STOP_LOSS_PCT = 0.02        
TAKE_PROFIT_PCT = 0.05

# === SIGNAL THRESHOLDS ===
BUY_THRESHOLD = 0.50        # Probability above this → BUY signal
SELL_THRESHOLD = 0.50        # Probability below this → SELL signal

# === LIVE LEARNING ===
RETRAIN_INTERVAL = 50        # Retrain model every N live samples
EWMA_ACCURACY_ALPHA = 0.1    # Weight for exponential moving average accuracy