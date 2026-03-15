import time
import pandas as pd
import config
from src.api_handler import APIHandler
from src.processor import DataProcessor
from src.ml_model import TradingModel
from src.risk_engine import RiskEngine

def run_live_bot():
    print("🚀 Initializing BYTE Algo Trading Sprint Bot...")
    
    # 1. Initialize all modules
    api = APIHandler(base_url=config.API_BASE_URL, api_key=config.API_KEY)
    processor = DataProcessor(target_col=config.TARGET_COL)
    model = TradingModel(model_path=config.MODEL_SAVE_PATH)
    
    risk = RiskEngine(
        starting_capital=config.STARTING_CAPITAL,
        risk_per_trade=config.RISK_PER_TRADE,
        stop_loss_pct=config.STOP_LOSS_PCT,
        take_profit_pct=config.TAKE_PROFIT_PCT
    )

    # Load the model trained on Day 1
    model.load_model()
    
    # Store recent data to calculate rolling features (Z-scores need history)
    history_buffer = []

    print("🟢 Bot is LIVE. Monitoring market stream...")

    while True:
        try:
            # 1. Fetch live market tick
            live_data = api.fetch_market_data()
            if not live_data:
                time.sleep(1)
                continue
                
            history_buffer.append(live_data)
            df = pd.DataFrame(history_buffer)

            # We need at least 30 rows to calculate our 30-period Z-scores/Bands
            if len(df) < 30:
                print(f"Buffering data... ({len(df)}/30)")
                time.sleep(1)
                continue

            # Keep buffer size manageable (e.g., last 100 ticks)
            if len(history_buffer) > 100:
                history_buffer.pop(0)

            # 2. Engineer Features (Z-scores, Bands, Variance)
            features_df = processor.engineer_features(df)
            
            # Extract just the latest row of features to make our prediction
            # Drop the target column since we don't know the future yet!
            latest_features = features_df.drop(columns=['target_up']).iloc[-1:]
            current_price = df.iloc[-1][config.TARGET_COL]

            # 3. Predict Probability
            up_prob = model.predict_prob(latest_features)

            # 4. Risk Engine Decision
            action, size = risk.decide(
                up_probability=up_prob, 
                current_price=current_price,
                buy_thresh=config.BUY_THRESHOLD,
                sell_thresh=config.SELL_THRESHOLD
            )

            # 5. Execute Trade
            if action in ["BUY", "SELL"]:
                print(f"⚡ {action} Signal! Prob: {up_prob:.2f} | Price: {current_price} | Size: {size}")
                api.execute_trade(action, size)
            else:
                print(f"⏸️ HOLD. Prob: {up_prob:.2f} | Price: {current_price}")

            # Match the competition's tick rate
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Bot manually stopped by team.")
            break
        except Exception as e:
            print(f"⚠️ Critical Error in loop: {e}. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run_live_bot()