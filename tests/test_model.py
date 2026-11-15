import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, make_supervised, chronological_split
from src.model import train_model
from src.evaluation import evaluate


def sample_data(n=40):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "solar": np.linspace(2.0, 6.0, n),
        "temperature": np.linspace(18.0, 35.0, n),
        "humidity": np.linspace(70.0, 35.0, n),
        "cloud": np.linspace(60.0, 10.0, n),
    }, index=dates)


def test_target_is_next_day_and_last_row_is_removed():
    raw = sample_data(10)
    prepared = make_supervised(raw)
    assert len(prepared) == 9
    assert prepared.iloc[0]["target"] == pytest.approx(raw.iloc[1]["solar"])
    assert prepared.iloc[0]["baseline"] == pytest.approx(raw.iloc[0]["solar"])
    assert prepared.iloc[0]["target_date"] == raw.index[0] + pd.Timedelta(days=1)


def test_missing_calendar_day_does_not_create_false_next_day_pair():
    raw = sample_data(12).drop(pd.Timestamp("2023-01-05"))
    prepared = make_supervised(raw)
    # Jan 4 cannot be used because Jan 5 target observation is missing.
    assert pd.Timestamp("2023-01-04") not in prepared.index
    # Jan 5 itself is absent; Jan 6 can still predict Jan 7.
    assert pd.Timestamp("2023-01-06") in prepared.index


def test_seasonal_features_are_finite():
    prepared = make_supervised(sample_data())
    assert np.isfinite(prepared[["day_sin", "day_cos"]].to_numpy()).all()


def test_chronological_split_does_not_shuffle():
    prepared = make_supervised(sample_data(50))
    train, test = chronological_split(prepared, test_fraction=0.2)
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(prepared)


def test_model_trains_and_predicts():
    prepared = make_supervised(sample_data(50))
    train, test = chronological_split(prepared)
    model = train_model(train)
    predictions = model.predict(test[FEATURES])
    assert len(predictions) == len(test)
    assert np.isfinite(predictions).all()


def test_evaluation_returns_required_metrics():
    prepared = make_supervised(sample_data(50))
    train, test = chronological_split(prepared)
    model = train_model(train)
    metrics = evaluate(test, model.predict(test[FEATURES]))
    assert {"model_mae", "model_rmse", "model_r2", "baseline_mae"} <= set(metrics)
    assert metrics["model_mae"] >= 0
    assert metrics["baseline_mae"] >= 0


def test_invalid_split_fraction_rejected():
    with pytest.raises(ValueError):
        chronological_split(make_supervised(sample_data()), 1.0)


def test_published_benchmark_artifacts_are_consistent():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    comparison = pd.read_csv(root / "results" / "solarsense_model_comparison.csv")
    summary = pd.read_csv(root / "results" / "solarsense_enhanced_summary.csv").iloc[0]
    predictions = pd.read_csv(root / "results" / "solarsense_enhanced_predictions.csv")

    assert len(predictions) == int(summary["test_observations"])
    assert comparison.iloc[0]["model"] == summary["selected_model"]
    assert mean_absolute_error(
        predictions["actual"], predictions["predicted"]
    ) == pytest.approx(summary["test_mae"], abs=1e-8)
    assert mean_absolute_error(
        predictions["actual"], predictions["persistence"]
    ) == pytest.approx(summary["persistence_mae"], abs=1e-8)
    assert predictions["target_date"].is_monotonic_increasing
