"""Unit tests for the pure-numpy quantile regression model (models.py)."""

import models
import numpy as np
import pytest


def test_r2_perfect_and_zero_variance():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert models._r2_score(y, y) == 1.0
    # Zero variance in y_true -> ss_tot == 0 -> guarded return of 0.0.
    const = np.array([5.0, 5.0, 5.0])
    assert models._r2_score(const, np.array([1.0, 2.0, 3.0])) == 0.0


def test_mae_and_rmse_known_values():
    assert models._mae(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])) == 1.0
    assert models._rmse(np.array([0.0, 0.0]), np.array([2.0, 2.0])) == 2.0


def test_irls_recovers_linear_relationship():
    x = np.linspace(0, 10, 200).reshape(-1, 1)
    y = 3.0 * x[:, 0] + 2.0
    coeffs = models._fit_quantile_irls(x, y, quantile=0.5)
    assert coeffs.shape == (2,)
    assert abs(coeffs[0] - 2.0) < 0.3
    assert abs(coeffs[1] - 3.0) < 0.3
    assert np.allclose(models._predict_quantile_irls(x, coeffs), y, atol=0.2)


def test_higher_quantile_predicts_higher_on_average():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, size=(500, 1))
    y = 2.0 * x[:, 0] + rng.normal(0, 1.0, size=500)
    p50 = models._predict_quantile_irls(x, models._fit_quantile_irls(x, y, 0.5))
    p90 = models._predict_quantile_irls(x, models._fit_quantile_irls(x, y, 0.9))
    assert p90.mean() > p50.mean()


def test_predict_before_train_raises():
    model = models.QuantileRegressionModel()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((3, 2)))


def test_train_predict_shape_and_coverage():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, size=(300, 2))
    y = x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.5, size=300)
    model = models.QuantileRegressionModel(quantile=0.75)
    model.train(x, y)
    preds = model.predict(x)
    assert preds.shape == (300,)
    assert 60.0 <= model.calculate_coverage(y, preds) <= 90.0
    assert set(model.evaluate(y, preds)) == {"r2", "mae", "rmse"}


def test_dynamic_peak_offpeak_routing():
    rng = np.random.default_rng(7)
    hours = np.tile(np.arange(24), 20)
    x = rng.uniform(0, 5, size=(len(hours), 2))
    peak = (hours >= 9) & (hours <= 22)
    y = x[:, 0] + np.where(peak, 5.0, 0.0) + rng.normal(0, 0.3, size=len(hours))
    config = {"peak_start": 9, "peak_end": 22, "peak_quantile": 0.75, "offpeak_quantile": 0.5}
    model = models.QuantileRegressionModel(dynamic_config=config)
    model.train(x, y, hours)
    assert model._coeffs_peak is not None
    assert model._coeffs_offpeak is not None
    preds = model.predict(x, hours)
    assert preds[peak].mean() > preds[~peak].mean() + 2.0


def test_dynamic_all_peak_leaves_offpeak_unset():
    rng = np.random.default_rng(3)
    hours = np.full(50, 12)
    x = rng.uniform(0, 5, size=(50, 1))
    y = x[:, 0] + rng.normal(0, 0.2, size=50)
    config = {"peak_start": 9, "peak_end": 22, "peak_quantile": 0.75, "offpeak_quantile": 0.5}
    model = models.QuantileRegressionModel(dynamic_config=config)
    model.train(x, y, hours)
    assert model._coeffs_peak is not None
    assert model._coeffs_offpeak is None
    assert model.predict(x, hours).shape == (50,)


def test_predict_iterative_no_power_lags_short_circuits():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 5, size=(20, 2))
    y = x[:, 0] + rng.normal(0, 0.3, size=20)
    model = models.QuantileRegressionModel(quantile=0.6)
    model.train(x, y)
    result = models.predict_iterative(x, np.zeros(20), model, ["hour", "temperature"], 0)
    assert result["iterative"] is False
    assert np.allclose(result["predictions"], model.predict(x))


def test_predict_iterative_feeds_predictions_into_power_lag():
    class Echo:
        """Stand-in model that echoes the power_lag_1 feature back as the prediction."""

        def predict(self, x, hours=None):
            return np.array([x[0, 0]])

    x = np.array([[7.0], [0.0], [0.0], [0.0]])
    result = models.predict_iterative(x, np.zeros(4), Echo(), ["power_lag_1"], 1)
    assert result["iterative"] is True
    # Each step writes the previous prediction into power_lag_1, so all equal the seed.
    assert np.allclose(result["predictions"], [7.0, 7.0, 7.0, 7.0])
