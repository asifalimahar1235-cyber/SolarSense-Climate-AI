"""Optional next-day demonstration using Open-Meteo weather forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .features import FEATURES

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_next_day_prediction(model, latitude=28.0167, longitude=69.3167):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,cloud_cover_mean",
        "timezone": "Asia/Karachi",
        "forecast_days": 2,
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=45)
    response.raise_for_status()
    daily = response.json()["daily"]

    # Today's forecast weather is used as the available input to predict tomorrow.
    input_date = pd.Timestamp(daily["time"][0])
    target_date = pd.Timestamp(daily["time"][1])
    row = pd.DataFrame([{
        "temperature": daily["temperature_2m_mean"][0],
        "humidity": daily["relative_humidity_2m_mean"][0],
        "cloud": daily["cloud_cover_mean"][0],
        "day_sin": np.sin(2 * np.pi * target_date.dayofyear / 365.25),
        "day_cos": np.cos(2 * np.pi * target_date.dayofyear / 365.25),
    }])[FEATURES]

    if row.isna().any().any():
        raise ValueError("Open-Meteo did not provide all required weather forecast inputs.")

    prediction = float(model.predict(row)[0])
    return {
        "input_date": input_date.date().isoformat(),
        "target_date": target_date.date().isoformat(),
        "predicted_solar": max(0.0, prediction),
    }
