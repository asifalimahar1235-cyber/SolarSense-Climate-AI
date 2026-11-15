"""Leakage-aware next-day target and feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["temperature", "humidity", "cloud", "day_sin", "day_cos"]


def make_supervised(frame: pd.DataFrame) -> pd.DataFrame:
    """Use weather on day t to predict solar radiation on day t+1.

    Seasonal features represent the target day (t+1), which is known in advance.
    The persistence baseline uses observed solar radiation on day t.
    """
    required = {"solar", "temperature", "humidity", "cloud"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    data = frame.copy().sort_index()
    target_dates = data.index + pd.Timedelta(days=1)
    day_number = target_dates.dayofyear

    supervised = pd.DataFrame(index=data.index)
    supervised["temperature"] = data["temperature"]
    supervised["humidity"] = data["humidity"]
    supervised["cloud"] = data["cloud"]
    supervised["day_sin"] = np.sin(2 * np.pi * day_number / 365.25)
    supervised["day_cos"] = np.cos(2 * np.pi * day_number / 365.25)
    supervised["target"] = data["solar"].shift(-1)
    supervised["baseline"] = data["solar"]
    supervised["target_date"] = target_dates
    supervised = supervised.dropna(subset=FEATURES + ["target", "baseline"])
    return supervised


def chronological_split(data: pd.DataFrame, test_fraction: float = 0.2):
    """Split rows in time order; do not shuffle."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if len(data) < 10:
        raise ValueError("At least 10 supervised rows are required.")
    split_at = int(len(data) * (1 - test_fraction))
    if split_at < 2 or len(data) - split_at < 2:
        raise ValueError("Not enough rows for both training and test sets.")
    return data.iloc[:split_at].copy(), data.iloc[split_at:].copy()
