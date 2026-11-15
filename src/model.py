"""Model training and prediction."""
from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def build_model() -> Pipeline:
    """Build a reproducible linear-regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])


def train_model(train):
    model = build_model()
    model.fit(train[FEATURES], train["target"])
    return model
