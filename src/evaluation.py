"""Evaluation utilities and failure analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(test: pd.DataFrame, predictions) -> dict:
    actual = test["target"].to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    baseline = test["baseline"].to_numpy(dtype=float)

    model_mae = float(mean_absolute_error(actual, predicted))
    baseline_mae = float(mean_absolute_error(actual, baseline))
    return {
        "model_mae": model_mae,
        "model_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "model_r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
        "baseline_mae": baseline_mae,
        "mae_improvement_percent": (
            float((baseline_mae - model_mae) / baseline_mae * 100)
            if baseline_mae > 0 else float("nan")
        ),
    }


def error_table(test: pd.DataFrame, predictions) -> pd.DataFrame:
    result = pd.DataFrame(index=test.index)
    result["target_date"] = test["target_date"]
    result["actual"] = test["target"].to_numpy()
    result["predicted"] = np.asarray(predictions)
    result["absolute_error"] = (result["actual"] - result["predicted"]).abs()
    return result.sort_values("absolute_error", ascending=False)
