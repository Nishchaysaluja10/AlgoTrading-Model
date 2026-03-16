import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.processor import DataProcessor

def analyze_features(model_path="models/xgb_model.pkl", data_path="data/historical_data.csv"):
    if not os.path.exists(model_path):
        print("❌ No trained model found.")
        return

    # 1. Load and Process data to get the EXACT feature names used in training
    df_raw = pd.read_csv(data_path)
    processor = DataProcessor()
    df_processed = processor.engineer_features(df_raw)
    
    # Remove the target column to match the training features
    X = df_processed.drop(columns=['target_up'])
    feature_names = X.columns.tolist()

    # 2. Load the model and extract averaged importance
    loaded_data = joblib.load(model_path)
    if isinstance(loaded_data, dict):
        ensemble = loaded_data['model']
        feature_names = loaded_data.get('feature_names', X.columns.tolist())
    else:
        ensemble = loaded_data
        feature_names = X.columns.tolist()
    
    # Extract importances from all models in the ensemble
    xgb_imp = ensemble.named_estimators_['xgb'].feature_importances_
    cat_imp = ensemble.named_estimators_['cat'].get_feature_importance()
    lgb_imp = ensemble.named_estimators_['lgb'].feature_importances_
    
    # Average them (Normalization might be needed, but simple average is often okay for ranking)
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