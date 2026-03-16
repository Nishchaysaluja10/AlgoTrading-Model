"""
Config — AlgoTrading Bot (Fixed)
"""
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# === SERVER CONNECTION ===
API_BASE_URL = os.getenv("API_BASE_URL", "http://YOUR_SERVER_IP:PORT/api")
API_KEY = os.getenv("API_KEY", "YOUR_API_KEY")

# === MODEL ===
TARGET_COL = "close"
MODEL_SAVE_PATH = "models/xgb_model.pkl"

# === RISK MANAGEMENT ===
STARTING_CAPITAL = 100000
RISK_PER_TRADE = 0.05        # 5% of capital per trade

STOP_LOSS_PCT = 0.015        # 1.5% stop loss (tighter = less bleed)
TAKE_PROFIT_PCT = 0.03       # 3% take profit (realistic for 2hr window)
TRAILING_STOP_PCT = 0.015    # 1.5% trailing stop to lock in profits

# === SIGNAL THRESHOLDS ===
# ORIGINAL: 0.65 / 0.35 — extreme asymmetry caused constant SELL bias
# FIXED:    0.55 / 0.45 — balanced, model only needs mild confidence
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

# === LIVE LEARNING ===
RETRAIN_INTERVAL = 50        # Check every 50 samples
EWMA_ACCURACY_ALPHA = 0.1    # Weight for exponential moving average