"""
Quantile Regression model for power consumption prediction.
Supports a fixed quantile, or per-sample (peak/off-peak) quantiles in a
single fit.

Pure numpy implementation — no scikit-learn dependency.
Uses Iteratively Reweighted Least Squares (IRLS) to solve the quantile
regression problem. This is the same underlying approach as sklearn's
QuantileRegressor but avoids the scipy.optimize.linprog LP solver, making
it significantly lighter and faster for small-to-medium datasets.

Algorithm overview (IRLS for quantile regression):
    The pinball (quantile) loss rho_q(r) is non-differentiable at r=0 but
    can be iteratively approximated as a weighted least squares problem:

        w_i = q_i / |r_i|      if r_i >= 0  (under-prediction, penalised by q_i)
        w_i = (1-q_i) / |r_i|  if r_i <  0  (over-prediction, penalised by 1-q_i)

    q_i may be one shared quantile or vary per sample (e.g. a higher, more
    conservative quantile during peak hours — still one coefficient set, so
    the fitted surface stays continuous across the window boundary).

    At each iteration we solve:
        (X'WX + reg) β = X'Wy

    where W = diag(w_i). The |r| in the weights is floored at 0.1% of the
    response dispersion (a Huber-smoothed pinball loss), and the L2 ridge is
    made dimensionless by the same dispersion (reg_j = n·α / (s·w_j²)), so the
    iteration is a monotone majorize-minimize descent on one fixed strictly
    convex objective, and a given alpha produces comparable smoothing across
    homes, units, and history lengths. Converges in typically 10–50
    iterations for datasets of this size.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core IRLS solver
# ---------------------------------------------------------------------------

def _fit_quantile_irls(
    X: np.ndarray,
    y: np.ndarray,
    quantile,
    alpha: float = 0.01,
    max_iter: int = 200,
    tol: float = 1e-6,
    reg_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fit a quantile regression model using IRLS.

    Args:
        X:        Feature matrix, shape (n_samples, n_features). Should be
                  pre-scaled if features are on different scales.
        y:        Target vector, shape (n_samples,).
        quantile: Target quantile in (0, 1) — a float, or an array of shape
                  (n_samples,) assigning each sample its own quantile (e.g. a
                  higher quantile for peak-hour samples). A per-sample quantile
                  changes only the loss asymmetry: there is still one
                  coefficient set, so predictions stay continuous in the
                  features with no regime boundaries.
        alpha:    L2 regularisation strength, applied to all coefficients
                  except the intercept. Anchored to the response dispersion
                  (see the objective note in the body), so it is a live,
                  monotone smoothing knob whose strength is comparable across
                  data scales and history lengths: ~0.1 smooths lightly,
                  ~1 visibly, ~3 strongly.
        max_iter: Maximum number of IRLS iterations.
        tol:      Convergence tolerance on max absolute coefficient change.
        reg_weights: Optional per-feature influence weights, length n_features.
                  When given, feature j's penalty becomes ``alpha / w_j**2`` — a
                  larger weight means a smaller penalty and therefore more
                  influence (equivalent, in predictions, to scaling column j by
                  w_j under a uniform penalty); a 0 weight drives the feature
                  to ~0 influence. ``None`` keeps the uniform alpha penalty.
                  Meaningful relative to each other on standardized features
                  with a non-trivial alpha.

    Returns:
        coeffs: 1-D array of length (n_features + 1). coeffs[0] is the
                intercept; coeffs[1:] are feature weights.
    """
    n, p = X.shape

    q = np.asarray(quantile, dtype=np.float64)
    if q.ndim > 0 and q.shape[0] != n:
        raise ValueError(
            f"per-sample quantile length {q.shape[0]} does not match "
            f"n_samples {n}"
        )

    # Augment X with a leading ones column for the intercept.
    Xa = np.empty((n, p + 1), dtype=np.float64)
    Xa[:, 0] = 1.0
    Xa[:, 1:] = X

    # -----------------------------------------------------------------------
    # Initialise with the ordinary least squares solution. This gives IRLS a
    # warm start that is usually close enough to converge in fewer iterations.
    # -----------------------------------------------------------------------
    coeffs, *_ = np.linalg.lstsq(Xa, y, rcond=None)

    # The solve minimizes a FIXED, strictly convex objective:
    #
    #     F(β) = Σ ρ_smooth,q(y - Xaβ)  +  ½ Σ_j reg_j β_j²
    #
    # where ρ_smooth is the pinball loss smoothed within ±floor (quadratic
    # inside the band, linear outside — the classic Huberized quantile loss),
    # and each IRLS iteration is a majorize-minimize step on F. Because the
    # objective never moves between iterations, the descent is monotone and
    # deterministic — no orbits, no path-dependent fits. Both the smoothing
    # band and the penalty are anchored to the RESPONSE DISPERSION
    #
    #     s = mean |y - median(y)|
    #
    # a fixed scale that never degenerates with fit quality:
    #   - floor = 1e-3·s caps any IRLS weight at ~1e3x the typical weight, so
    #     the handful of samples the fit interpolates can no longer swamp the
    #     ridge (which silently disabled alpha and the influence weights), and
    #     the quantile is biased only within ~0.1% of the target's spread;
    #   - reg_j = n·alpha / (s·w_j²) makes the penalty dimensionless: the
    #     weighted Gram of a standardized column scales like n/s, so the
    #     shrinkage a given alpha produces is comparable across homes, units,
    #     and history lengths. alpha stays a live, monotone, portable knob.
    # A larger influence weight w_j means a smaller penalty and therefore
    # more influence; the small floor on w_j turns a 0 weight into a very
    # large (finite) penalty, driving the feature to ~0 influence without an
    # inf/NaN in the normal equations.
    _eps = 1e-8
    y_scale = float(np.mean(np.abs(y - np.median(y))))
    if y_scale < _eps:
        y_scale = max(float(np.mean(np.abs(y))), 1.0)
    floor = max(_eps, 1e-3 * y_scale)

    reg_diag = np.full(p + 1, n * alpha / y_scale)
    reg_diag[0] = 0.0
    if reg_weights is not None:
        w = np.maximum(np.abs(np.asarray(reg_weights, dtype=np.float64)), 1e-6)
        reg_diag[1:] = n * alpha / (y_scale * w ** 2)

    for iteration in range(max_iter):
        coeffs_prev = coeffs.copy()

        # Residuals under the current estimate.
        residuals = y - Xa @ coeffs

        # Asymmetric IRLS weights derived from the pinball loss subgradient
        # (the exact MM majorizer of the smoothed pinball). q broadcasts
        # whether it is a scalar or a per-sample vector.
        abs_res = np.maximum(np.abs(residuals), floor)
        weights = np.where(residuals >= 0,
                           q / abs_res,
                           (1.0 - q) / abs_res)

        # Weighted normal equations:  (Xa' W Xa + reg) β = Xa' W y
        # Multiply each row of Xa by its weight without forming a dense W matrix.
        XaW = Xa.T * weights          # shape (p+1, n)
        A = XaW @ Xa                  # shape (p+1, p+1)
        A.flat[:: p + 2] += reg_diag  # add regularisation to diagonal in-place
        b = XaW @ y                   # shape (p+1,)

        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Singular system — fall back to least-squares pseudo-inverse.
            coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

        # Convergence check.
        if np.max(np.abs(coeffs - coeffs_prev)) < tol:
            logger.debug("IRLS converged after %d iterations", iteration + 1)
            break
    else:
        logger.debug("IRLS reached max_iter=%d without converging", max_iter)

    return coeffs


def _predict_quantile_irls(X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Predict using IRLS-fitted coefficients.

    Args:
        X:      Feature matrix, shape (n_samples, n_features).
        coeffs: Coefficient vector from _fit_quantile_irls (intercept first).

    Returns:
        Predicted values, shape (n_samples,).
    """
    n = X.shape[0]
    Xa = np.empty((n, coeffs.shape[0]), dtype=np.float64)
    Xa[:, 0] = 1.0
    Xa[:, 1:] = X
    return Xa @ coeffs


# ---------------------------------------------------------------------------
# Pure-numpy metric helpers  (replaces sklearn.metrics imports)
# ---------------------------------------------------------------------------

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Public model class  — same interface as the original sklearn-backed version
# ---------------------------------------------------------------------------

class QuantileRegressionModel:
    """
    Quantile Regression model for conservative power forecasting.

    Drop-in replacement for the previous sklearn-based implementation.
    Internally uses IRLS (Iteratively Reweighted Least Squares) with pure
    numpy — no scipy or scikit-learn required.
    """

    def __init__(
        self,
        quantile: float = 0.75,
        dynamic_config: Optional[Dict] = None,
        alpha: float = 0.01,
        feature_weights: Optional[np.ndarray] = None,
        standardize: bool = False,
    ):
        """
        Args:
            quantile: Quantile to predict (0.5–0.99). Used when
                ``dynamic_config`` is not given (or no hours are supplied).
            dynamic_config: Optional dict for peak/off-peak modelling:
                - peak_start:       Hour when peak period begins (e.g. 9)
                - peak_end:         Hour when peak period ends   (e.g. 22)
                - peak_quantile:    Quantile for peak hours      (e.g. 0.75)
                - offpeak_quantile: Quantile for off-peak hours  (e.g. 0.50)
                Implemented as ONE fit with a per-sample quantile (each
                training sample is assigned its window's quantile inside the
                pinball loss). A single coefficient set means the prediction
                surface is continuous — no separate sub-models, no
                discontinuity at the window boundaries, and no risk of an
                unfittable (empty) window.
            alpha: L2 regularisation strength passed to the IRLS solver.
                Anchored to the response dispersion, so it is a live,
                monotone smoothing knob with comparable strength across
                homes and history lengths: ~0.1 light, ~1 visible, ~3 strong.
            feature_weights: Optional per-feature influence weights (length =
                n_features), applied as the per-feature ridge penalty
                ``alpha / w_j**2``. Use with ``standardize=True`` and a
                non-trivial ``alpha`` for a meaningful, comparable effect.
            standardize: When True, z-score the feature matrix on the training
                statistics and reuse those stats at predict time. This makes
                the penalty (and the influence weights) act uniformly across
                features and conditions the solve.
        """
        self.quantile = quantile
        self.dynamic_config = dynamic_config
        self.alpha = alpha
        self.feature_weights = (
            None if feature_weights is None
            else np.asarray(feature_weights, dtype=np.float64)
        )
        self.standardize = standardize

        # Standardization stats, fit on the training matrix (shared by the
        # peak/off-peak sub-models so the future frame's scaling matches both).
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

        # Fitted state: one coefficient vector from _fit_quantile_irls (also
        # for peak/off-peak fits — the window asymmetry lives in the loss, not
        # in separate coefficient sets).
        self._coeffs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------

    def _fit_scaler(self, X: np.ndarray) -> None:
        """Fit standardization statistics on the training matrix (once)."""
        if not self.standardize:
            self._mu = None
            self._sigma = None
            return
        self._mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        # Constant columns (e.g. 'year' over a short window) have ~0 std; a
        # floor of 1.0 keeps them at zero post-centering instead of exploding.
        sigma[sigma < 1e-8] = 1.0
        self._sigma = sigma

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """Standardize a feature matrix with the stored training statistics."""
        if not self.standardize or self._mu is None:
            return X
        return (X - self._mu) / self._sigma

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hours_train: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the quantile regression model.

        Args:
            X_train:     Training feature matrix.
            y_train:     Training target vector.
            hours_train: Hour-of-day values, required for dynamic quantile.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        if (
            self.feature_weights is not None
            and self.feature_weights.shape[0] != X_train.shape[1]
        ):
            raise ValueError(
                f"feature_weights length {self.feature_weights.shape[0]} does not "
                f"match feature count {X_train.shape[1]}"
            )

        # Fit the scaler on the FULL training matrix, then standardize once.
        self._fit_scaler(X_train)
        Xs = self._apply_scaler(X_train)

        if self.dynamic_config is not None and hours_train is not None:
            # Peak/off-peak in ONE fit: each sample gets its window's quantile
            # inside the pinball loss. The hour features (not separate
            # coefficient sets) carry the level difference between windows, so
            # the fitted surface has no boundary discontinuity.
            peak_start = self.dynamic_config['peak_start']
            peak_end = self.dynamic_config['peak_end']
            hours_arr = np.asarray(hours_train)
            peak_mask = (hours_arr >= peak_start) & (hours_arr <= peak_end)
            q_vec = np.where(
                peak_mask,
                self.dynamic_config['peak_quantile'],
                self.dynamic_config['offpeak_quantile'],
            )
            print(f"  - Training Quantile Regression (single fit, IRLS): "
                  f"q={self.dynamic_config['peak_quantile']:.2f} on "
                  f"{int(peak_mask.sum())} peak samples ({peak_start}–{peak_end}), "
                  f"q={self.dynamic_config['offpeak_quantile']:.2f} on "
                  f"{int((~peak_mask).sum())} off-peak samples")
            self._coeffs = _fit_quantile_irls(
                Xs, y_train, q_vec, self.alpha,
                reg_weights=self.feature_weights,
            )
        else:
            print(f"  - Training Quantile Regression (q={self.quantile:.2f}, IRLS)...")
            self._coeffs = _fit_quantile_irls(
                Xs, y_train, self.quantile, self.alpha,
                reg_weights=self.feature_weights,
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        hours: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict power consumption.

        Args:
            X:     Feature matrix.
            hours: Accepted for API compatibility; unused. The peak/off-peak
                   quantile asymmetry is baked into the coefficients at
                   training time (per-sample quantiles), so prediction is one
                   continuous surface — no hour-based routing, and therefore
                   no discontinuity at the window boundaries.

        Returns:
            Predicted values, shape (n_samples,).
        """
        # Standardize with the stored TRAINING stats (no-op when standardize is
        # off). Done here, on the raw matrix, so predict_iterative can keep
        # writing raw-kW predictions back into the power-lag columns without
        # any unit mismatch — the scaling is applied per call, after the lag
        # write-backs, inside this method.
        X = self._apply_scaler(np.asarray(X, dtype=np.float64))
        if self._coeffs is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return _predict_quantile_irls(X, self._coeffs)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute regression metrics.

        Returns:
            Dict with keys 'r2', 'mae', 'rmse'.
        """
        return {
            'r2':   _r2_score(y_true, y_pred),
            'mae':  _mae(y_true, y_pred),
            'rmse': _rmse(y_true, y_pred),
        }

    def calculate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Percentage of actual values at or below the predicted quantile.

        Returns:
            Coverage as a value in [0, 100].
        """
        return float(np.mean(y_true <= y_pred) * 100)


# ---------------------------------------------------------------------------
# Iterative forecasting  — identical interface to original
# ---------------------------------------------------------------------------

def predict_iterative(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: QuantileRegressionModel,
    features: list,
    n_power_lags: int,
    hours_test: Optional[np.ndarray] = None,
    state_model: Optional[QuantileRegressionModel] = None,
) -> Dict[str, Any]:
    """
    Perform iterative (auto-regressive) prediction, feeding a predicted value
    back as the lag feature for subsequent steps.

    Simulates real-time forecasting where future actual power values are
    unknown. Falls back to a single vectorised call when no power lag
    features are present.

    State vs reported value (``state_model``)
    -----------------------------------------
    The value written back into the ``power_lag_*`` columns and the value
    reported are two different roles. By default (``state_model=None``) both are
    the same — ``model``'s own prediction is fed back. But ``model`` predicts a
    high quantile (e.g. peak q=0.75), and recursively feeding a high quantile
    back into its own lags makes the conservative margin **compound** over the
    horizon (the forecast drifts up and never settles to the overnight trough).
    Quantiles do not propagate linearly through the AR recursion.

    When ``state_model`` is given (a median / q=0.5 model trained on the same
    features), the lag columns are seeded from *its* prediction — the conditional
    median, a stable central estimate — while the **reported** value still comes
    from ``model`` (the conservative quantile). This decouples the AR state from
    the reported quantile: the trajectory propagates at the median (no upward
    drift) but each reported hour keeps its quantile margin.

    Args:
        X_test:       Test feature matrix.
        y_test:       Test target vector (used for evaluation by caller).
        model:        Trained QuantileRegressionModel (reported quantile).
        features:     List of feature names corresponding to columns of X_test.
        n_power_lags: Number of power lag features in the model.
        hours_test:   Hour-of-day values for dynamic quantile (optional). Passed
                      to both ``model`` and ``state_model``.
        state_model:  Optional median model whose predictions seed the lag
                      columns. ``None`` feeds ``model``'s own predictions back
                      (legacy behaviour).

    Returns:
        Dict with keys:
            'predictions': np.ndarray of predicted (reported) values.
            'iterative':   bool — True if auto-regressive loop was used.
    """
    n_samples = len(X_test)
    predictions = np.zeros(n_samples, dtype=np.float64)
    # Values fed into the lag columns of later rows. Equals ``predictions`` when
    # no state_model is given; otherwise the median model's stable estimate.
    lag_values = np.zeros(n_samples, dtype=np.float64)

    # Indices of power-lag columns inside the feature matrix.
    power_lag_indices = [i for i, feat in enumerate(features) if 'power_lag' in feat]

    if not power_lag_indices:
        # No lag features — predict in one vectorised pass.
        return {
            'predictions': model.predict(X_test, hours_test),
            'iterative': False,
        }

    # Auto-regressive loop: update lag columns from the previous state values.
    for i in range(n_samples):
        X_current = X_test[i: i + 1].copy()

        if i > 0:
            for lag_order, feat_idx in enumerate(power_lag_indices, start=1):
                if i >= lag_order:
                    X_current[0, feat_idx] = lag_values[i - lag_order]

        hour_current = None if hours_test is None else hours_test[i: i + 1]
        predictions[i] = model.predict(X_current, hour_current)[0]
        lag_values[i] = (
            predictions[i] if state_model is None
            else state_model.predict(X_current, hour_current)[0]
        )

    return {
        'predictions': predictions,
        'iterative': True,
    }
