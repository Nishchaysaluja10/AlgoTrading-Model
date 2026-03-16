"""
Main — Live Trading Bot
Connects to the real trading server, runs predictions, and tracks accuracy.
"""
import time
import pandas as pd
import config
from src.api_handler import APIHandler
from src.processor import DataProcessor
from src.ml_model import TradingModel

def run_live_bot():
    print("🚀 Initializing AlgoTrading Bot...")
    print(f"📡 Connecting to: {config.API_BASE_URL}")
    
    # 1. Initialize all modules
    api = APIHandler(base_url=config.API_BASE_URL, api_key=config.API_KEY)
    processor = DataProcessor(target_col=config.TARGET_COL)
    model = TradingModel(model_path=config.MODEL_SAVE_PATH)
    
    # Load the pre-trained model
    model.load_model()
    
    # --- REAL-TIME DATA HANDLING ---
    history_buffer = []          # Ticks used for feature engineering
    live_learning_samples = []   # Pairs of (features, outcome) to retrain later
    last_features = None         # Features from the previous tick
    last_features_price = None   # Price when last features were extracted
    last_prob = None             # Prediction from the previous tick
    
    # --- ACCURACY TRACKING ---
    predictions_history = []     # List of bools: was prediction correct?
    correct_count = 0
    ewma_accuracy = 0.5          # Start at 50% (no bias)
    alpha = config.EWMA_ACCURACY_ALPHA

    print("🟢 Bot is LIVE. Monitoring market stream...")

    while True:
        try:
            # 1. Fetch live market tick
            live_data = api.fetch_market_data()
            if not live_data:
                time.sleep(1)
                continue
                
            # Current price from this tick
            current_price = live_data[config.TARGET_COL]

            # --- LIVE PERFORMANCE & LEARNING LOGIC ---
            if last_features is not None and last_prob is not None:
                # 1. Evaluate previous prediction
                actual_up = 1 if current_price > last_features_price else 0
                
                # Was our last prediction correct?
                predicted_up = 1 if last_prob > 0.5 else 0
                is_correct = (predicted_up == actual_up)
                predictions_history.append(is_correct)
                
                if is_correct: correct_count += 1
                
                # Simple running accuracy
                raw_accuracy = (correct_count / len(predictions_history)) * 100
                
                # EWMA accuracy (weights recent predictions more)
                ewma_accuracy = alpha * (1.0 if is_correct else 0.0) + (1 - alpha) * ewma_accuracy
                ewma_pct = ewma_accuracy * 100
                
                print(f"📊 Live Accuracy: {raw_accuracy:.2f}% (EWMA: {ewma_pct:.1f}%) | Sample Size: {len(predictions_history)}")

                # 2. Store for retraining
                sample = last_features.copy()
                sample['target_up'] = actual_up
                live_learning_samples.append(sample)
                
                # Auto-Retrain at configurable interval with minimum sample check
                if (len(live_learning_samples) >= 30 and 
                    len(live_learning_samples) % config.RETRAIN_INTERVAL == 0):
                    print(f"🧠 Refocusing model on {len(live_learning_samples)} live samples...")
                    live_df = pd.DataFrame(live_learning_samples)
                    recent_df = live_df.tail(config.RETRAIN_INTERVAL * 2)
                    model.train(recent_df.drop(columns=['target_up']), recent_df['target_up'])

            # Update buffer for feature engineering
            history_buffer.append(live_data)
            df = pd.DataFrame(history_buffer)

            # We need at least 30 rows to calculate our 30-period indicators
            if len(df) < 30:
                print(f"Buffering data... ({len(df)}/30)")
                time.sleep(1)
                continue

            # Keep a larger buffer for indicator stability
            if len(history_buffer) > 200:
                history_buffer.pop(0)

            # 2. Engineer Features
            features_df = processor.engineer_features(df, training=False)
            
            if len(features_df) == 0:
                print("⚠️ Not enough data for all features yet, waiting...")
                time.sleep(1)
                continue
            
            # Extract just the latest row of features
            latest_features_row = features_df.iloc[-1:]
            
            # Store for the NEXT tick's target calculation
            last_features = latest_features_row.to_dict('records')[0]
            last_features_price = current_price

            # 3. Predict Probability
            up_prob = model.predict_prob(latest_features_row)
            last_prob = up_prob

            # 4. Decision logic based on probability
            if up_prob >= config.BUY_THRESHOLD:
                action = "BUY"
            elif up_prob <= config.SELL_THRESHOLD:
                action = "SELL"
            else:
                action = "HOLD"

            # 5. Confidence assessment
            confidence = abs(up_prob - 0.5) * 200  # 0-100% scale
            conf_label = "🔥HIGH" if confidence > 30 else "📉LOW"

            # 6. Output Prediction
            if action in ["BUY", "SELL"]:
                print(f"⚡ {action} Signal! Prob: {up_prob:.2f} | Conf: {conf_label} ({confidence:.0f}%) | Price: {current_price}")
            else:
                print(f"⏸️ HOLD. Prob: {up_prob:.2f} | Conf: {conf_label} ({confidence:.0f}%) | Price: {current_price}")

            # Match the server's tick rate
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Bot stopped.")
            if predictions_history:
                final_acc = (correct_count / len(predictions_history)) * 100
                print(f"📊 Final Accuracy: {final_acc:.2f}% over {len(predictions_history)} predictions")
                print(f"📊 Final EWMA Accuracy: {ewma_accuracy * 100:.1f}%")
            break
        except Exception as e:
            print(f"⚠️ Error in loop: {e}. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run_live_bot()