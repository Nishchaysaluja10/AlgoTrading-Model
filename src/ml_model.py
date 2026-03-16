import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import os
import pandas as pd

class TradingModel:
    def __init__(self, model_path="models/xgb_model.pkl"):
        self.model_path = model_path
        
        # 1. XGBoost: Good for general patterns
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,              # Reduced from 6 to curb overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,       # Prevents splits on tiny leaf groups
            reg_alpha=0.1,            # L1 regularization
            eval_metric='logloss',
            random_state=42
        )
        
        # 2. CatBoost: Robust to overfitting and handles categorical features well
        cat_model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=5,                  # Reduced from 6
            l2_leaf_reg=5,
            min_data_in_leaf=10,      # Prevents overfitting on small samples
            verbose=0,
            random_state=42
        )

        # 3. LightGBM: Fast and often finds different patterns than XGBoost
        lgb_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,              # Added explicit depth limit
            subsample=0.8,
            colsample_bytree=0.8,
            min_data_in_leaf=10,      # Prevents overfitting on small samples
            random_state=42,
            verbose=-1
        )
        
        # 4. The Voting Committee (Ensemble of 3 different architectures)
        self.model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model), 
                ('cat', cat_model),
                ('lgb', lgb_model)
            ],
            voting='soft',
            weights=[2, 2, 1]  # XGB+Cat stronger on small datasets; LGB as tiebreaker
        )
        self.is_trained = False
        self.feature_names = None

    def train(self, X_train, y_train):
        print("🧠 Training Ensemble Committee (XGB + Cat + LGB)...")
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.save_model()

    def predict_prob(self, features):
        if not self.is_trained:
            self.load_model()
        
        # Enforce feature order if it's a DataFrame
        if isinstance(features, pd.DataFrame) and self.feature_names:
            features = features[self.feature_names]
        
        # predict_proba returns [prob_class_0, prob_class_1]
        prob = self.model.predict_proba(features)
        return prob[0][1]

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # Save both the model and the feature names
        data_to_save = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        joblib.dump(data_to_save, self.model_path)
        print(f"✅ Ensemble Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            loaded_data = joblib.load(self.model_path)
            if isinstance(loaded_data, dict):
                self.model = loaded_data['model']
                self.feature_names = loaded_data.get('feature_names')
            else:
                self.model = loaded_data
                self.feature_names = None
            self.is_trained = True
            print(f"✅ Ensemble Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"❌ Model file {self.model_path} not found. Run train.py first.")