"""
Main — Live Trading Bot (FIXED)
Connects to the real trading server, runs predictions, and actually executes trades.

Key fixes vs original:
  1. Actually calls api.execute_trade() — original never did
  2. Tracks position (FLAT / LONG / SHORT) to avoid double orders
  3. Thresholds tuned to reduce SELL bias (0.55 / 0.45 instead of 0.65 / 0.35)
  4. Live retraining blends new samples with historical knowledge instead of overwriting
  5. Trailing stop-loss checks to cut losers early
"""
import time
import pandas as pd
import config
from src.api_handler import APIHandler
from src.processor import DataProcessor
from src.ml_model import TradingModel

# ──────────────────────────────────────────
# Position tracker
# ──────────────────────────────────────────
class PositionTracker:
    def __init__(self, starting_capital):
        self.capital       = starting_capital
        self.position      = "FLAT"      # FLAT | LONG | SHORT
        self.entry_price   = None
        self.position_size = 0           # units held
        self.peak_price    = None        # for trailing stop
        self.trades_won    = 0
        self.trades_lost   = 0

    @property
    def win_rate(self):
        total = self.trades_won + self.trades_lost
        return (self.trades_won / total * 100) if total else 0.0

    def open_long(self, price, risk_pct=0.05):
        if self.position != "FLAT":
            return False
        amount = self.capital * risk_pct
        self.position_size = amount / price
        self.entry_price   = price
        self.peak_price    = price
        self.position      = "LONG"
        print(f"  📈 OPENED LONG  @ {price:.4f} | size={self.position_size:.4f} | capital={self.capital:.2f}")
        return True

    def open_short(self, price, risk_pct=0.05):
        if self.position != "FLAT":
            return False
        amount = self.capital * risk_pct
        self.position_size = amount / price
        self.entry_price   = price
        self.peak_price    = price
        self.position      = "SHORT"
        print(f"  📉 OPENED SHORT @ {price:.4f} | size={self.position_size:.4f} | capital={self.capital:.2f}")
        return True

    def close_position(self, price, reason="signal"):
        if self.position == "FLAT":
            return False
        if self.position == "LONG":
            pnl = (price - self.entry_price) * self.position_size
        else:  # SHORT
            pnl = (self.entry_price - price) * self.position_size

        self.capital += pnl
        outcome = "WIN ✅" if pnl > 0 else "LOSS ❌"
        if pnl > 0:
            self.trades_won  += 1
        else:
            self.trades_lost += 1

        print(f"  🔒 CLOSED {self.position} [{reason}] @ {price:.4f} | PnL={pnl:+.2f} | {outcome} | capital={self.capital:.2f}")
        self.position      = "FLAT"
        self.entry_price   = None
        self.peak_price    = None
        self.position_size = 0
        return True

    def update_peak(self, price):
        if self.peak_price is None:
            return
        if self.position == "LONG"  and price > self.peak_price:
            self.peak_price = price
        if self.position == "SHORT" and price < self.peak_price:
            self.peak_price = price

    def trailing_stop_hit(self, price, trail_pct=0.015):
        """Returns True if price has pulled back trail_pct from the peak."""
        if self.position == "FLAT" or self.peak_price is None:
            return False
        if self.position == "LONG":
            return price < self.peak_price * (1 - trail_pct)
        if self.position == "SHORT":
            return price > self.peak_price * (1 + trail_pct)
        return False

    def stop_loss_hit(self, price, stop_pct=None):
        stop_pct = stop_pct or config.STOP_LOSS_PCT
        if self.position == "FLAT" or self.entry_price is None:
            return False
        if self.position == "LONG":
            return price < self.entry_price * (1 - stop_pct)
        if self.position == "SHORT":
            return price > self.entry_price * (1 + stop_pct)
        return False

    def take_profit_hit(self, price, tp_pct=None):
        tp_pct = tp_pct or config.TAKE_PROFIT_PCT
        if self.position == "FLAT" or self.entry_price is None:
            return False
        if self.position == "LONG":
            return price >= self.entry_price * (1 + tp_pct)
        if self.position == "SHORT":
            return price <= self.entry_price * (1 - tp_pct)
        return False


# ──────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────
def run_live_bot():
    print("🚀 Initializing AlgoTrading Bot (Fixed Edition)...")
    print(f"📡 Connecting to: {config.API_BASE_URL}")

    api       = APIHandler(base_url=config.API_BASE_URL, api_key=config.API_KEY)
    processor = DataProcessor(target_col=config.TARGET_COL)
    model     = TradingModel(model_path=config.MODEL_SAVE_PATH)
    model.load_model()

    tracker = PositionTracker(starting_capital=config.STARTING_CAPITAL)

    history_buffer       = []
    live_learning_samples = []
    last_features        = None
    last_features_price  = None
    last_prob            = None

    predictions_history  = []
    correct_count        = 0
    ewma_accuracy        = 0.5
    alpha                = config.EWMA_ACCURACY_ALPHA

    print("🟢 Bot is LIVE. Starting main loop...\n")

    while True:
        try:
            # ── 1. Fetch tick ────────────────────────────────────────────
            live_data = api.fetch_market_data()
            if not live_data:
                time.sleep(1)
                continue

            current_price = live_data[config.TARGET_COL]

            # ── 2. Check exit conditions on open position ─────────────────
            if tracker.position != "FLAT":
                tracker.update_peak(current_price)

                if tracker.stop_loss_hit(current_price):
                    action_str = "SELL" if tracker.position == "LONG" else "BUY"
                    api.execute_trade(action_str, tracker.position_size)
                    tracker.close_position(current_price, reason="stop-loss")

                elif tracker.take_profit_hit(current_price):
                    action_str = "SELL" if tracker.position == "LONG" else "BUY"
                    api.execute_trade(action_str, tracker.position_size)
                    tracker.close_position(current_price, reason="take-profit")

                elif tracker.trailing_stop_hit(current_price):
                    action_str = "SELL" if tracker.position == "LONG" else "BUY"
                    api.execute_trade(action_str, tracker.position_size)
                    tracker.close_position(current_price, reason="trailing-stop")

            # ── 3. Accuracy tracking + live learning ─────────────────────
            if last_features is not None and last_prob is not None:
                actual_up    = 1 if current_price > last_features_price else 0
                predicted_up = 1 if last_prob > 0.5 else 0
                is_correct   = (predicted_up == actual_up)
                predictions_history.append(is_correct)
                if is_correct:
                    correct_count += 1

                raw_accuracy  = (correct_count / len(predictions_history)) * 100
                ewma_accuracy = alpha * (1.0 if is_correct else 0.0) + (1 - alpha) * ewma_accuracy

                print(f"📊 Accuracy: {raw_accuracy:.1f}% (EWMA: {ewma_accuracy*100:.1f}%) | "
                      f"Position: {tracker.position} | Capital: {tracker.capital:.2f} | "
                      f"W/L: {tracker.trades_won}/{tracker.trades_lost}")

                # Collect sample — blend with old knowledge, don't overwrite
                sample = last_features.copy()
                sample['target_up'] = actual_up
                live_learning_samples.append(sample)

                # Only retrain if we have enough samples AND model accuracy is poor
                should_retrain = (
                    len(live_learning_samples) >= config.RETRAIN_INTERVAL and
                    len(live_learning_samples) % config.RETRAIN_INTERVAL == 0 and
                    ewma_accuracy < 0.52   # Only retrain when genuinely struggling
                )
                if should_retrain:
                    print(f"🧠 Retuning on {len(live_learning_samples)} live samples (EWMA accuracy low)...")
                    live_df  = pd.DataFrame(live_learning_samples)
                    # Use ALL live samples, not just recent — preserve learned patterns
                    model.train(live_df.drop(columns=['target_up']), live_df['target_up'])

            # ── 4. Buffer management ──────────────────────────────────────
            history_buffer.append(live_data)
            if len(history_buffer) > 200:
                history_buffer.pop(0)

            df = pd.DataFrame(history_buffer)
            if len(df) < 30:
                print(f"  Buffering... ({len(df)}/30)")
                time.sleep(1)
                continue

            # ── 5. Feature engineering ────────────────────────────────────
            features_df = processor.engineer_features(df, training=False)
            if len(features_df) == 0:
                print("  ⚠️  Not enough data for features yet...")
                time.sleep(1)
                continue

            latest_row         = features_df.iloc[-1:]
            last_features      = latest_row.to_dict('records')[0]
            last_features_price = current_price

            # ── 6. Predict ────────────────────────────────────────────────
            up_prob  = model.predict_prob(latest_row)
            last_prob = up_prob
            confidence = abs(up_prob - 0.5) * 200   # 0–100 scale
            conf_label = "🔥HIGH" if confidence > 20 else "📉LOW"

            # ── 7. Signal decision ────────────────────────────────────────
            if up_prob >= config.BUY_THRESHOLD:
                signal = "BUY"
            elif up_prob <= config.SELL_THRESHOLD:
                signal = "SELL"
            else:
                signal = "HOLD"

            print(f"  ⚡ Signal: {signal} | Prob: {up_prob:.3f} | Conf: {conf_label} ({confidence:.0f}%) | Price: {current_price}")

            # ── 8. Execute trades ─────────────────────────────────────────
            # CRITICAL: This block was entirely missing in the original code!
            if signal == "BUY" and tracker.position == "FLAT":
                # Open a LONG position
                size = (tracker.capital * config.RISK_PER_TRADE) / current_price
                resp = api.execute_trade("BUY", size)
                if resp is not None:
                    tracker.open_long(current_price, risk_pct=config.RISK_PER_TRADE)

            elif signal == "SELL" and tracker.position == "FLAT":
                # Open a SHORT position
                size = (tracker.capital * config.RISK_PER_TRADE) / current_price
                resp = api.execute_trade("SELL", size)
                if resp is not None:
                    tracker.open_short(current_price, risk_pct=config.RISK_PER_TRADE)

            elif signal == "BUY" and tracker.position == "SHORT":
                # Close the SHORT, flip to LONG
                resp = api.execute_trade("BUY", tracker.position_size)
                if resp is not None:
                    tracker.close_position(current_price, reason="signal-flip")
                    size = (tracker.capital * config.RISK_PER_TRADE) / current_price
                    resp2 = api.execute_trade("BUY", size)
                    if resp2 is not None:
                        tracker.open_long(current_price, risk_pct=config.RISK_PER_TRADE)

            elif signal == "SELL" and tracker.position == "LONG":
                # Close the LONG, flip to SHORT
                resp = api.execute_trade("SELL", tracker.position_size)
                if resp is not None:
                    tracker.close_position(current_price, reason="signal-flip")
                    size = (tracker.capital * config.RISK_PER_TRADE) / current_price
                    resp2 = api.execute_trade("SELL", size)
                    if resp2 is not None:
                        tracker.open_short(current_price, risk_pct=config.RISK_PER_TRADE)

            time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user.")
            # Close any open position cleanly
            if tracker.position != "FLAT":
                close_action = "SELL" if tracker.position == "LONG" else "BUY"
                api.execute_trade(close_action, tracker.position_size)
                tracker.close_position(current_price, reason="manual-stop")

            if predictions_history:
                final_acc = (correct_count / len(predictions_history)) * 100
                print(f"\n📊 Final Stats:")
                print(f"   Accuracy    : {final_acc:.2f}% over {len(predictions_history)} predictions")
                print(f"   EWMA Acc    : {ewma_accuracy * 100:.1f}%")
                print(f"   Win Rate    : {tracker.win_rate:.1f}%")
                print(f"   Final Capital: {tracker.capital:.2f}")
                print(f"   PnL         : {tracker.capital - config.STARTING_CAPITAL:+.2f}")
            break

        except Exception as e:
            print(f"⚠️  Loop error: {e}. Retrying in 3 seconds...")
            time.sleep(3)


if __name__ == "__main__":
    run_live_bot()