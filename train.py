"""
Train — Train the model on historical data before going live.

The evaluation is chronological (the final 20% is never used for fitting) and
reports directional accuracy, Sharpe, total return, and maximum drawdown both
before and after the confidence/volatility risk filters.

Usage: python train.py
"""
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import config
from backtest import evaluate_predictions
from src.processor import DataProcessor
from src.ml_model import TradingModel


def run_training(data_path='data/asset_alpha_training.csv'):
    print("🚀 Starting Model Training Phase...")
    try:
        raw_df = pd.read_csv(data_path)
        raw_df.columns = raw_df.columns.str.lower()
        print(f"✅ Loaded raw data: {raw_df.shape} | Columns: {list(raw_df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: Place your historical CSV at '{data_path}' first.")
        print("   Expected columns: open, high, low, close, volume")
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
    X, y = processed_df[feature_cols], processed_df['target_up']

    up_count = int(y.sum())
    down_count = int(len(y) - up_count)
    ratio = float(down_count) / float(up_count + 1e-8)
    print(f"\n📊 Class balance — UP: {up_count} ({up_count/len(y)*100:.1f}%) | DOWN: {down_count} ({down_count/len(y)*100:.1f}%)")

    split_idx = int(len(processed_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"📈 Training on {len(X_train)} rows, Testing on {len(X_test)} rows...\n")

    model = TradingModel(model_path=config.MODEL_SAVE_PATH, class_ratio=ratio)
    model.train(X_train, y_train)
    predictions = model.model.predict(X_test)
    probs = model.model.predict_proba(X_test)[:, 1]

    print("\n🧠 Evaluation on unseen chronological test data")
    print(f"🎯 Directional Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"📊 Probability distribution — Mean: {probs.mean():.3f} | Std: {probs.std():.3f}")

    # Test returns are aligned to the feature rows; use the original close series
    # so the report includes an explicit, reproducible backtest result.
    test_prices = raw_df['close'].iloc[-len(processed_df):].iloc[split_idx:].reset_index(drop=True)
    future_returns = test_prices.shift(-1).div(test_prices).sub(1)
    valid = future_returns.notna()
    volatility = future_returns.rolling(5).std()
    metrics = evaluate_predictions(pd.Series(probs)[valid], y_test.reset_index(drop=True)[valid],
                                   future_returns[valid], volatility[valid])
    with open('backtest_results.json', 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
        handle.write('\n')

    print("\n📈 BACKTEST RESULTS")
    print(f"   Directional accuracy : {metrics['directional_accuracy_pct']:.2f}%")
    for label in ('before_filters', 'after_filters'):
        result = metrics[label]
        print(f"   {label.replace('_', ' ').title():<18}: Sharpe {result['sharpe']:.4f} | "
              f"Max drawdown {result['max_drawdown_pct']:.2f}% | "
              f"Return {result['total_return_pct']:.2f}% | Trades {result['trades']}")
    print("✅ Metrics saved to backtest_results.json")


if __name__ == "__main__":
    run_training()
