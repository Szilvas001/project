"""Tests for HistoricalGHITrainer — validates R² ≥ 75% and RMSE < 10%."""

import numpy as np
import pandas as pd
import pytest

from solar_forecast.allsky.historical_trainer import (
    AccuracyTargetNotMet,
    HistoricalGHITrainer,
    TrainingResult,
    build_features,
    synthesize_training_data,
)


def _make_trainer():
    return HistoricalGHITrainer(n_estimators=50, random_state=0)


# ── synthesize_training_data ──────────────────────────────────────────────

def test_synthesize_returns_dataframe():
    df = synthesize_training_data(n_days=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10 * 24
    assert {"ghi_clear", "cloud_cover", "ghi_obs"}.issubset(df.columns)


def test_synthesize_ghi_obs_non_negative():
    df = synthesize_training_data(n_days=10)
    assert (df["ghi_obs"] >= 0).all()


# ── build_features ────────────────────────────────────────────────────────

def test_build_features_adds_cyclic_columns():
    df = synthesize_training_data(n_days=3)
    out = build_features(df)
    for col in ("hour_sin", "hour_cos", "doy_sin", "doy_cos"):
        assert col in out.columns, f"missing {col}"


def test_build_features_missing_ghi_clear_raises():
    df = pd.DataFrame({"cloud_cover": [0.5]})
    with pytest.raises(ValueError, match="ghi_clear"):
        build_features(df)


# ── HistoricalGHITrainer.train_and_validate ───────────────────────────────

def test_accuracy_contract_met():
    """Core contract: R² ≥ 0.75 and RMSE_rel ≤ 0.10 on synthetic data."""
    df = synthesize_training_data(n_days=90, seed=42)
    trainer = _make_trainer()
    result = trainer.train_and_validate(df, val_fraction=0.25, enforce=True)
    assert isinstance(result, TrainingResult)
    assert result.r2 >= 0.75, f"R² too low: {result.r2:.3f}"
    assert result.rmse_relative <= 0.10, f"RMSE_rel too high: {result.rmse_relative:.3f}"


def test_accuracy_result_fields():
    df = synthesize_training_data(n_days=30, seed=1)
    trainer = _make_trainer()
    result = trainer.train_and_validate(df, enforce=False)
    assert result.n_train > 0
    assert result.n_val > 0
    assert result.mae >= 0
    assert isinstance(result.feature_importance, dict)


def test_enforce_false_doesnt_raise_on_small_data():
    """A tiny dataset might underperform; enforce=False must not raise."""
    df = synthesize_training_data(n_days=2, seed=99)
    trainer = _make_trainer()
    result = trainer.train_and_validate(df, enforce=False)
    assert result.r2 is not None  # computed but not enforced


# ── predict ───────────────────────────────────────────────────────────────

def test_predict_shape_matches_input():
    df = synthesize_training_data(n_days=14, seed=7)
    trainer = _make_trainer()
    trainer.fit(df)
    preds = trainer.predict(df)
    assert preds.shape == (len(df),)


def test_predict_non_negative():
    df = synthesize_training_data(n_days=14, seed=7)
    trainer = _make_trainer()
    trainer.fit(df)
    preds = trainer.predict(df)
    assert (preds >= 0).all()


def test_predict_before_fit_raises():
    trainer = _make_trainer()
    df = synthesize_training_data(n_days=3)
    with pytest.raises(RuntimeError):
        trainer.predict(df)


# ── save / load ───────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    df = synthesize_training_data(n_days=20, seed=5)
    trainer = _make_trainer()
    trainer.fit(df)
    preds_before = trainer.predict(df)

    path = tmp_path / "ghi_model.joblib"
    trainer.save(path)
    loaded = HistoricalGHITrainer.load(path)
    preds_after = loaded.predict(df)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


def test_save_before_fit_raises(tmp_path):
    trainer = _make_trainer()
    with pytest.raises(RuntimeError):
        trainer.save(tmp_path / "model.joblib")
