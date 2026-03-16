"""
Analyze — Visualize feature importances from the trained ensemble model.
Usage: python analyze.py
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import config
from src.processor import DataProcessor

def analyze_features():
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("❌ No trained model found. Run train.py first.")
        return

    # 1. Load and process data to get the EXACT feature names used in training
    df_raw = pd.read_csv(config.DATA_PATH)
    processor = DataProcessor(target_col=config.TARGET_COL, volume_col='volume')
    df_processed = processor.engineer_features(df_raw)
    
    # Remove the target column to match the training features
    X = df_processed.drop(columns=['target_up'])
    feature_names = X.columns.tolist()

    # 2. Load the model and extract averaged importance
    loaded_data = joblib.load(config.MODEL_SAVE_PATH)
    if isinstance(loaded_data, dict):
        ensemble = loaded_data['model']
        feature_names = loaded_data.get('feature_names', feature_names)
    else:
        ensemble = loaded_data
    
    # Extract importances from all models in the ensemble
    xgb_imp = ensemble.named_estimators_['xgb'].feature_importances_
    cat_imp = ensemble.named_estimators_['cat'].get_feature_importance()
    lgb_imp = ensemble.named_estimators_['lgb'].feature_importances_
    
    # Average them
    importances = (xgb_imp + cat_imp + lgb_imp) / 3

    # 3. Match and Plot
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 8))
    feat_imp.tail(15).plot(kind='barh', color='teal')
    plt.title("Top 15 Most Influential Features (Ensemble Average)")
    plt.xlabel("Importance Score")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig("feature_importance.png")
    print("✅ Analysis complete! 'feature_importance.png' has been generated.")
    plt.show()

if __name__ == "__main__":
    analyze_features()