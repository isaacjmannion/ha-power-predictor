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


def test_predict_iterative_state_model_seeds_lags_not_reported_value():
    """With a state_model, lags are seeded from IT (the median), not the
    reported value — so the reported trajectory follows the state, decoupling
    the AR state from the conservative quantile (no upward compounding)."""

    class Echo:
        """Reports the power_lag_1 feature it is given."""

        def predict(self, x, hours=None):
            return np.array([x[0, 0]])

    class ConstState:
        """Stand-in median model: always feeds 2.0 into the lags."""

        def predict(self, x, hours=None):
            return np.array([2.0])

    x = np.array([[7.0], [0.0], [0.0], [0.0]])
    result = models.predict_iterative(
        x, np.zeros(4), Echo(), ["power_lag_1"], 1, state_model=ConstState()
    )
    # Step 0 reports the 7.0 seed but feeds the state's 2.0 forward; every later
    # step then sees 2.0 in its lag. (Legacy self-feedback would give all 7.0.)
    assert np.allclose(result["predictions"], [7.0, 2.0, 2.0, 2.0])


# --- Per-feature ridge weighting -------------------------------------------

def test_reg_weights_of_one_reproduce_uniform_alpha():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, size=(200, 3))
    y = x @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 0.3, size=200)
    uniform = models._fit_quantile_irls(x, y, 0.5, alpha=1.0)
    weighted = models._fit_quantile_irls(x, y, 0.5, alpha=1.0, reg_weights=np.ones(3))
    assert np.allclose(uniform, weighted, atol=1e-10)


def test_higher_weight_increases_feature_coefficient():
    # On standardized features with a non-trivial alpha, raising a feature's
    # weight lowers its penalty and lets its coefficient grow.
    rng = np.random.default_rng(12)
    x = rng.normal(0, 1, size=(400, 2))
    y = 3.0 * x[:, 0] + 1.0 * x[:, 1] + rng.normal(0, 0.5, size=400)
    low = models._fit_quantile_irls(x, y, 0.5, alpha=5.0, reg_weights=np.array([0.25, 1.0]))
    high = models._fit_quantile_irls(x, y, 0.5, alpha=5.0, reg_weights=np.array([4.0, 1.0]))
    # coeffs[1] is the first feature (intercept is coeffs[0]).
    assert abs(high[1]) > abs(low[1])


def test_reg_weights_via_penalty_equivalent_to_column_scaling():
    # reg_diag[j] = alpha / w_j**2 must match scaling column j by w_j under a
    # uniform penalty — to floating-point precision.
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, size=(150, 2))
    y = x @ np.array([2.0, -1.0]) + rng.normal(0, 0.2, size=150)
    w = 3.0

    via_penalty = models._fit_quantile_irls(
        x, y, 0.5, alpha=1.0, reg_weights=np.array([w, 1.0])
    )
    x_scaled = x.copy()
    x_scaled[:, 0] *= w
    via_scaling = models._fit_quantile_irls(x_scaled, y, 0.5, alpha=1.0)

    pred_penalty = models._predict_quantile_irls(x, via_penalty)
    # Undo the scaling on the prediction side: feed the scaled matrix back.
    pred_scaling = models._predict_quantile_irls(x_scaled, via_scaling)
    assert np.allclose(pred_penalty, pred_scaling, atol=1e-8)


# --- Standardization --------------------------------------------------------

def test_standardize_predict_matches_manual_transform():
    rng = np.random.default_rng(14)
    # Columns on very different scales — the case standardization exists for.
    x = np.column_stack([
        rng.uniform(2020, 2026, size=300),
        rng.uniform(0, 23, size=300),
        rng.uniform(-5, 40, size=300),
    ])
    y = 0.1 * x[:, 1] + 0.05 * x[:, 2] + rng.normal(0, 0.5, size=300)
    model = models.QuantileRegressionModel(quantile=0.5, standardize=True)
    model.train(x, y)
    preds = model.predict(x)
    # Manually applying the stored scaler and the raw solver must agree.
    xs = (x - model._mu) / model._sigma
    manual = models._predict_quantile_irls(xs, model._coeffs)
    assert np.allclose(preds, manual, atol=1e-9)
    assert np.isfinite(preds).all()


def test_standardize_handles_constant_column():
    rng = np.random.default_rng(15)
    x = np.column_stack([
        np.full(120, 2026.0),            # constant 'year'-like column
        rng.uniform(0, 10, size=120),
    ])
    y = x[:, 1] + rng.normal(0, 0.3, size=120)
    model = models.QuantileRegressionModel(quantile=0.5, standardize=True)
    model.train(x, y)
    preds = model.predict(x)
    assert np.isfinite(preds).all()
    # sigma floor keeps the constant column from producing inf/nan.
    assert model._sigma[0] == 1.0


def test_feature_weights_length_mismatch_raises():
    model = models.QuantileRegressionModel(
        quantile=0.5, standardize=True, feature_weights=np.ones(5)
    )
    with pytest.raises(ValueError):
        model.train(np.zeros((10, 3)), np.zeros(10))


def test_standardized_dynamic_model_runs_iteratively_with_power_lags():
    rng = np.random.default_rng(16)
    hours = np.tile(np.arange(24), 15)
    n = len(hours)
    # Two features: a power lag and temperature, on different scales.
    power_lag = rng.uniform(0.5, 6.0, size=n)
    temp = rng.uniform(-5, 35, size=n)
    x = np.column_stack([power_lag, temp])
    y = 0.8 * power_lag + 0.02 * temp + rng.normal(0, 0.2, size=n)
    config = {"peak_start": 9, "peak_end": 22, "peak_quantile": 0.75, "offpeak_quantile": 0.5}
    model = models.QuantileRegressionModel(
        dynamic_config=config,
        alpha=1.0,
        feature_weights=np.array([2.0, 0.5]),
        standardize=True,
    )
    model.train(x, y, hours)
    result = models.predict_iterative(
        x, np.zeros(n), model, ["power_lag_1", "temperature"], 1, hours_test=hours
    )
    assert result["iterative"] is True
    assert result["predictions"].shape == (n,)
    assert np.isfinite(result["predictions"]).all()
