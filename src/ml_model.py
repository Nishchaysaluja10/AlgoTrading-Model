import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import os
import pandas as pd

class TradingModel:
    def __init__(self, model_path="models/xgb_model.pkl", class_ratio=1.0):
        """
        class_ratio: down_count / up_count from training data.
        Passing this into XGBoost/LightGBM as scale_pos_weight corrects
        the SELL bias that appears when historical data has more down-moves.
        """
        self.model_path  = model_path
        self.class_ratio = class_ratio   # used for class-balancing

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            scale_pos_weight=class_ratio,   # ← KEY FIX: balances class weight
            eval_metric='logloss',
            random_state=42
        )

        cat_model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=5,
            min_data_in_leaf=10,
            # FIX: class_weights with np.float64 breaks sklearn clone().
            # auto_class_weights lets CatBoost compute balance from data itself.
            auto_class_weights='Balanced',
            verbose=0,
            random_state=42
        )

        lgb_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_data_in_leaf=10,
            scale_pos_weight=class_ratio,      # ← balances class weight
            random_state=42,
            verbose=-1
        )

        self.model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('cat', cat_model),
                ('lgb', lgb_model)
            ],
            voting='soft',
            weights=[2, 2, 1]
        )
        self.is_trained    = False
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
        if isinstance(features, pd.DataFrame) and self.feature_names:
            features = features[self.feature_names]
        prob = self.model.predict_proba(features)
        return prob[0][1]

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        data_to_save = {
            'model': self.model,
            'feature_names': self.feature_names,
            'class_ratio': self.class_ratio,
        }
        joblib.dump(data_to_save, self.model_path)
        print(f"✅ Ensemble Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            loaded_data = joblib.load(self.model_path)
            if isinstance(loaded_data, dict):
                self.model        = loaded_data['model']
                self.feature_names = loaded_data.get('feature_names')
                self.class_ratio  = loaded_data.get('class_ratio', 1.0)
            else:
                self.model        = loaded_data
                self.feature_names = None
            self.is_trained = True
            print(f"✅ Ensemble Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(
                f"❌ Model file {self.model_path} not found. Run train.py first."
            )