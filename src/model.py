"""
Model Training, Loading, and Prediction Wrapper
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from src.config import RANDOM_SEED, CLASS_NAMES, FINAL_MODEL_PATH
from src.features import extract_features

class CyberSentinelModel:
    def __init__(self, n_estimators: int = 350, max_depth: int = 16, min_samples_split: int = 4):
        self.class_names = CLASS_NAMES
        self.label_to_id = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.id_to_label = {i: c for i, c in enumerate(CLASS_NAMES)}
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names = []

    def fit(self, df: pd.DataFrame):
        X_df = extract_features(df)
        self.feature_names = list(X_df.columns)
        X = X_df.values
        y = np.array([self.label_to_id[l] for l in df['label']])
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() or load() first.")
        X_df = extract_features(df)
        return self.model.predict_proba(X_df.values)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(df)
        pred_ids = np.argmax(probs, axis=1)
        return np.array([self.id_to_label[i] for i in pred_ids])

    def save(self, path: str = FINAL_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, path)

    @classmethod
    def load(cls, path: str = FINAL_MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}. Please train first.")
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.class_names = data['class_names']
        instance.feature_names = data['feature_names']
        instance.is_fitted = data['is_fitted']
        return instance
