"""
Train — Train the model on historical data before going live.

Fix added: scale_pos_weight / class_weight balancing to prevent SELL bias
when historical data has more down-moves than up-moves.

Usage: python train.py
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import config
from src.processor import DataProcessor
from src.ml_model import TradingModel

def run_training(data_path='data/asset_alpha_training.csv'):
    print("🚀 Starting Model Training Phase...")

    try:
        raw_df = pd.read_csv(data_path)
        # FIX 1: Normalize column names to lowercase.
        # CSV has Open/High/Low/Close/Volume (capitalized) but the processor
        # expects lowercase. Without this, candle/ATR/volume features all silently fail.
        raw_df.columns = raw_df.columns.str.lower()
        print(f"✅ Loaded raw data: {raw_df.shape} | Columns: {list(raw_df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: Place your historical CSV at '{data_path}' first.")
        print(f"   Expected columns: open, high, low, close, volume")
        return

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in raw_df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return

    processor = DataProcessor(target_col=config.TARGET_COL, volume_col='volume')
    processed_df = processor.engineer_features(raw_df)

    ignore_cols = ['target_up', 'timestamp', 'time', 'date', 'id']
    feature_cols = [col for col in processed_df.columns if col.lower() not in ignore_cols]

    X = processed_df[feature_cols]
    y = processed_df['target_up']

    # ── Class balance check ───────────────────────────────────────────────
    up_count   = int(y.sum())
    down_count = int(len(y) - up_count)
    # FIX 2: Cast to plain Python float — np.float64 breaks sklearn's clone()
    # inside VotingClassifier.fit(), causing the CatBoost RuntimeError.
    ratio = float(down_count) / float(up_count + 1e-8)
    print(f"\n📊 Class balance — UP: {up_count} ({up_count/len(y)*100:.1f}%) | DOWN: {down_count} ({down_count/len(y)*100:.1f}%)")
    if ratio > 1.3 or ratio < 0.77:
        print(f"  ⚠️  Imbalanced dataset (ratio={ratio:.2f}). scale_pos_weight will compensate.")
    else:
        print(f"  ✅ Dataset is well balanced (ratio={ratio:.2f}).")
    # ─────────────────────────────────────────────────────────────────────

    split_idx = int(len(processed_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"📈 Training on {len(X_train)} rows, Testing on {len(X_test)} rows...\n")

    model = TradingModel(model_path=config.MODEL_SAVE_PATH, class_ratio=ratio)
    model.train(X_train, y_train)

    print("\n🧠 Evaluating on unseen test data...")
    predictions = model.model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"🎯 Test Accuracy: {acc * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, predictions))

    # Show predicted probability distribution — should be spread, not all <0.4
    probs = model.model.predict_proba(X_test)[:, 1]
    print(f"\n📊 Probability distribution on test set:")
    # FIX 3: typo 'npobs' → 'probs'
    print(f"   Mean: {probs.mean():.3f} | Std: {probs.std():.3f}")
    print("\n✅ Training complete! Model saved and ready for live trading.")
    print(f"   % above 0.55: {(probs > 0.55).mean()*100:.1f}%  ← These become BUY")
    print(f"   % below 0.45: {(probs < 0.45).mean()*100:.1f}%  ← These become SELL")
    print(f"   % in 0.45-0.55: {((probs >= 0.45) & (probs <= 0.55)).mean()*100:.1f}%  ← HOLD zone")


if __name__ == "__main__":
    run_training()