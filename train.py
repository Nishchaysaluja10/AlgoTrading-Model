import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import config
from src.processor import DataProcessor
from src.ml_model import TradingModel

def run_training():
    print("🚀 Starting Day 1 Model Training Phase...")

    # 1. Load historical data (Update this filename tomorrow!)
    data_path = 'data/historical_data.csv' 
    try:
        raw_df = pd.read_csv(data_path)
        print(f"✅ Loaded raw data: {raw_df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Place the competition CSV at '{data_path}' first.")
        return

    # 2. Engineer Features
    processor = DataProcessor(target_col=config.TARGET_COL, volume_col='volume')
    processed_df = processor.engineer_features(raw_df)

    # 3. Prepare X (Features) and y (Target)
    # Automatically ignore non-math columns so XGBoost doesn't crash
    ignore_cols = ['target_up', 'timestamp', 'time', 'date', 'id']
    feature_cols = [col for col in processed_df.columns if col.lower() not in ignore_cols]
    
    X = processed_df[feature_cols]
    y = processed_df['target_up']

    # 4. Strict Chronological Split (80% Train, 20% Test)
    split_idx = int(len(processed_df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"📈 Training on {len(X_train)} rows, Testing on {len(X_test)} rows...")

    # 5. Train and Save Model
    model = TradingModel(model_path=config.MODEL_SAVE_PATH)
    model.train(X_train, y_train)

    # 6. Evaluate
    print("🧠 Evaluating Model Performance on unseen Test Data...")
    predictions = model.model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")
    
    # Classification report tells you if the bot is better at finding BUYs or SELLs
    print("\n📊 Classification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    run_training()