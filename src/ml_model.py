import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import os

class TradingModel:
    def __init__(self, model_path="models/xgb_model.pkl"):
        self.model_path = model_path
        
        # 1. The original XGBoost Brain
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        )
        
        # 2. The New CatBoost Brain (Built to strictly prevent overfitting)
        cat_model = CatBoostClassifier(
            iterations=150,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3, # Adds a mathematical penalty for overly complex rules
            verbose=0,     # Keeps the terminal output clean
            random_state=42
        )
        
        # 3. The Voting Committee ('soft' averages their probabilities)
        self.model = VotingClassifier(
            estimators=[('xgb', xgb_model), ('cat', cat_model)],
            voting='soft',
            weights=[1,2]
        )
        self.is_trained = False

    def train(self, X_train, y_train):
        print("🧠 Training XGBoost + CatBoost Ensemble Committee...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.save_model()

    def predict_prob(self, features):
        if not self.is_trained:
            self.load_model()
        
        # predict_proba returns [prob_class_0, prob_class_1]
        prob = self.model.predict_proba(features)
        return prob[0][1]

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"✅ Ensemble Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            print(f"✅ Ensemble Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"❌ Model file {self.model_path} not found. Run train.py first.")