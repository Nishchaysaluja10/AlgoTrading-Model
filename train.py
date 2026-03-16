"""
Train — Train the model on historical data before going live.
Usage: python train.py
"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import config
from src.processor import DataProcessor
from src.ml_model import TradingModel

def run_training(data_path='data/historical_data.csv'):
    print("🚀 Starting Model Training Phase...")

    # 1. Load historical data
    try:
        raw_df = pd.read_csv(data_path)
        print(f"✅ Loaded raw data: {raw_df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Place your historical CSV at '{data_path}' first.")
        print(f"   Expected columns: timestamp, open, high, low, close, volume")
        return

    # 2. Validate required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in raw_df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        print(f"   Your CSV has: {list(raw_df.columns)}")
        return

    # 3. Engineer Features
    processor = DataProcessor(target_col=config.TARGET_COL, volume_col='volume')
    processed_df = processor.engineer_features(raw_df)

    # 4. Prepare X (Features) and y (Target)
    ignore_cols = ['target_up', 'timestamp', 'time', 'date', 'id']
    feature_cols = [col for col in processed_df.columns if col.lower() not in ignore_cols]
    
    X = processed_df[feature_cols]
    y = processed_df['target_up']

    # 5. Strict Chronological Split (80% Train, 20% Test)
    split_idx = int(len(processed_df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"📈 Training on {len(X_train)} rows, Testing on {len(X_test)} rows...")

    # 6. Train and Save Model
    model = TradingModel(model_path=config.MODEL_SAVE_PATH)
    model.train(X_train, y_train)

    # 7. Evaluate
    print("🧠 Evaluating Model Performance on unseen Test Data...")
    predictions = model.model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    run_training()