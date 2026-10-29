import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class XGBoostClassificationHead:
    """
    XGBoost Classifier.
    Used as the final Decision Layer in the multimodal pipeline.
    Replaces the standard Dense NN head for higher accuracy and better tabular-like 
    splitting on the dense feature vectors outputted by the Fusion Attention model.
    """
    def __init__(self, **kwargs):
        # Default hyperparams tuned for generic deep representations
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(kwargs)
        self.model = xgb.XGBClassifier(**default_params)
        self.is_trained = False
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None):
        """
        Trains the XGBoost model on the extracted feature vectors.
        X_train: [num_samples, feature_dim]
        y_train: [num_samples] indicating 0 (Healthy) or 1 (Infected)
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        print("Training XGBoost Classifier on dense features...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_trained = True
        print("XGBoost training complete.")
        
    def predict(self, X_test: np.ndarray):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict().")
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test: np.ndarray):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict_proba().")
        return self.model.predict_proba(X_test)[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        preds = self.predict(X_test)
        probs = self.predict_proba(X_test)
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        report = classification_report(y_test, preds)
        
        return {
            'accuracy': acc,
            'auc': auc,
            'report': report
        }

if __name__ == "__main__":
    # Test XGBoost head shape compatibility
    X_mock = np.random.rand(100, 128)
    y_mock = np.random.randint(0, 2, size=100)
    
    xgb_head = XGBoostClassificationHead()
    xgb_head.train(X_mock, y_mock)
    metrics = xgb_head.evaluate(X_mock, y_mock)
    print(f"XGBoost Test AUC: {metrics['auc']:.4f}")
