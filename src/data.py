"""Download and validate daily climate observations from NASA POWER."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

LATITUDE = 28.0167
LONGITUDE = 69.3167
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "CLOUD_AMT"]
COLUMN_MAP = {
    "ALLSKY_SFC_SW_DWN": "solar",
    "T2M": "temperature",
    "RH2M": "humidity",
    "CLOUD_AMT": "cloud",
}


def fetch_nasa_power(
    start: str = "20150101",
    end: Optional[str] = None,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    timeout: int = 120,
) -> pd.DataFrame:
    """Fetch daily NASA POWER data. Dates must be YYYYMMDD.

    Missing observations are retained as NaN on a complete daily index. This is
    important: dropping missing dates before target shifting could accidentally
    turn a multi-day gap into a false next-day training example.
    """
    if end is None:
        end = (datetime.now(timezone.utc).date() - timedelta(days=7)).strftime("%Y%m%d")

    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    response = requests.get(NASA_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    try:
        values = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise ValueError("NASA POWER response did not contain the expected parameter data.") from exc

    missing = [name for name in PARAMETERS if name not in values]
    if missing:
        raise ValueError(f"NASA POWER response is missing parameters: {missing}")

    frame = pd.DataFrame(values)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d", errors="coerce")
    frame.index.name = "date"
    frame = frame.rename(columns=COLUMN_MAP)
    frame = frame[["solar", "temperature", "humidity", "cloud"]]
    frame = frame.apply(pd.to_numeric, errors="coerce").replace(-999, float("nan"))
    frame = frame.sort_index()
    frame = frame[~frame.index.isna()]

    if frame.empty:
        raise ValueError("NASA POWER returned no observations for the requested period.")

    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.asfreq("D")

    valid_solar = frame["solar"].dropna()
    if (valid_solar < 0).any():
        raise ValueError("Solar radiation contains negative values after missing-data cleaning.")
    return frame


def save_dataset(frame: pd.DataFrame, path: str = "data/ghotki_nasa_power.csv") -> str:
    """Save daily observations to CSV, creating the parent directory."""
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index_label="date")
    return str(destination)
