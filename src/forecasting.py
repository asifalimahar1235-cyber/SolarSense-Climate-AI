"""Experimental next-available-day estimate using the benchmark feature design.

This is a current-data demonstration, not the historical holdout evaluation.
It trains the benchmark's Extra Trees configuration on all available labelled
NASA POWER observations and predicts the day after the latest usable input day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

FEATURES = [
    "solar_lag_1",
    "solar_lag_2",
    "solar_lag_3",
    "solar_roll_3",
    "solar_roll_7",
    "temperature",
    "humidity",
    "cloud",
    "temperature_change_1d",
    "cloud_change_1d",
    "target_day_sin",
    "target_day_cos",
]


def build_forecast_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the same day-t inputs used by the enhanced benchmark."""
    required = {"solar", "temperature", "humidity", "cloud"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")
    data = frame.copy().sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input data must use a DatetimeIndex.")
    if data.index.has_duplicates:
        raise ValueError("Input dates must be unique.")
    data = data.asfreq("D")

    target_dates = data.index + pd.Timedelta(days=1)
    day_number = target_dates.dayofyear
    features = pd.DataFrame(index=data.index)
    features["solar_lag_1"] = data["solar"]
    features["solar_lag_2"] = data["solar"].shift(1)
    features["solar_lag_3"] = data["solar"].shift(2)
    features["solar_roll_3"] = data["solar"].rolling(3, min_periods=3).mean()
    features["solar_roll_7"] = data["solar"].rolling(7, min_periods=7).mean()
    features["temperature"] = data["temperature"]
    features["humidity"] = data["humidity"]
    features["cloud"] = data["cloud"]
    features["temperature_change_1d"] = data["temperature"].diff()
    features["cloud_change_1d"] = data["cloud"].diff()
    features["target_day_sin"] = np.sin(2 * np.pi * day_number / 365.25)
    features["target_day_cos"] = np.cos(2 * np.pi * day_number / 365.25)
    return features


def forecast_next_available_day(frame: pd.DataFrame) -> dict:
    """Fit the benchmark Extra Trees configuration and estimate next available day."""
    features = build_forecast_features(frame)
    training = features.join(frame.sort_index().asfreq("D")["solar"].shift(-1).rename("target"))
    training = training.dropna(subset=FEATURES + ["target"])
    forecast_inputs = features.dropna(subset=FEATURES)

    if len(training) < 100:
        raise ValueError("At least 100 complete training examples are required.")
    if forecast_inputs.empty:
        raise ValueError("There are not enough recent complete observations to forecast.")

    latest = forecast_inputs.iloc[[-1]]
    model = ExtraTreesRegressor(
        n_estimators=250,
        min_samples_leaf=3,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(training[FEATURES], training["target"])
    prediction = max(0.0, float(model.predict(latest[FEATURES])[0]))
    input_date = latest.index[0]
    return {
        "input_date": input_date.date().isoformat(),
        "target_date": (input_date + pd.Timedelta(days=1)).date().isoformat(),
        "predicted_solar": prediction,
        "training_observations": int(len(training)),
        "latest_observation": float(latest["solar_lag_1"].iloc[0]),
    }
