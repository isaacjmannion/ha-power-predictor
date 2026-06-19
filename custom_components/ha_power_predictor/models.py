"""
Quantile Regression model for power consumption prediction.
Supports both standard and dynamic peak/off-peak quantile modeling.

Pure numpy implementation — no scikit-learn dependency.
Uses Iteratively Reweighted Least Squares (IRLS) to solve the quantile
regression problem. This is the same underlying approach as sklearn's
QuantileRegressor but avoids the scipy.optimize.linprog LP solver, making
it significantly lighter and faster for small-to-medium datasets.

Algorithm overview (IRLS for quantile regression):
    The pinball (quantile) loss rho_q(r) is non-differentiable at r=0 but
    can be iteratively approximated as a weighted least squares problem:

        w_i = q / |r_i|      if r_i >= 0  (under-prediction, penalised by q)
        w_i = (1-q) / |r_i|  if r_i <  0  (over-prediction, penalised by 1-q)

    At each iteration we solve:
        (X'WX + αI) β = X'Wy

    where W = diag(w_i) and α is L2 regularisation. This converges in
    typically 20–50 iterations for datasets of this size.
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
    quantile: float,
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
        quantile: Target quantile in (0, 1).
        alpha:    L2 regularisation strength (matches sklearn's alpha=0.01
                  default). Applied to all coefficients except the intercept.
        max_iter: Maximum number of IRLS iterations.
        tol:      Convergence tolerance on max absolute coefficient change.
        reg_weights: Optional per-feature influence weights, length n_features.
                  When given, feature j's penalty becomes ``alpha / w_j**2`` — a
                  larger weight means a smaller penalty and therefore more
                  influence (equivalent, in predictions, to scaling column j by
                  w_j under a uniform penalty). ``None`` keeps the uniform alpha
                  penalty. Only meaningful when X is standardized and alpha is
                  non-trivial.

    Returns:
        coeffs: 1-D array of length (n_features + 1). coeffs[0] is the
                intercept; coeffs[1:] are feature weights.
    """
    n, p = X.shape

    # Augment X with a leading ones column for the intercept.
    Xa = np.empty((n, p + 1), dtype=np.float64)
    Xa[:, 0] = 1.0
    Xa[:, 1:] = X

    # -----------------------------------------------------------------------
    # Initialise with the ordinary least squares solution. This gives IRLS a
    # warm start that is usually close enough to converge in fewer iterations.
    # -----------------------------------------------------------------------
    coeffs, *_ = np.linalg.lstsq(Xa, y, rcond=None)

    # Regularisation diagonal: skip the intercept term (index 0). With
    # per-feature weights, feature j's penalty is alpha / w_j**2 (a larger
    # weight -> smaller penalty -> more influence). A small floor on the weight
    # turns a 0 weight into a very large (finite) penalty, so the feature is
    # driven to ~0 influence without an inf/NaN in the normal equations.
    reg_diag = np.full(p + 1, alpha)
    reg_diag[0] = 0.0
    if reg_weights is not None:
        w = np.maximum(np.abs(np.asarray(reg_weights, dtype=np.float64)), 1e-6)
        reg_diag[1:] = alpha / (w ** 2)

    # A small floor prevents division-by-zero when a residual is exactly 0.
    _eps = 1e-8

    for iteration in range(max_iter):
        coeffs_prev = coeffs.copy()

        # Residuals under the current estimate.
        residuals = y - Xa @ coeffs

        # Asymmetric IRLS weights derived from the pinball loss subgradient.
        abs_res = np.maximum(np.abs(residuals), _eps)
        weights = np.where(residuals >= 0,
                           quantile / abs_res,
                           (1.0 - quantile) / abs_res)

        # Weighted normal equations:  (Xa' W Xa + αI) β = Xa' W y
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
            quantile: Quantile to predict (0.5–0.99).
            dynamic_config: Optional dict for peak/off-peak modelling:
                - peak_start:       Hour when peak period begins (e.g. 9)
                - peak_end:         Hour when peak period ends   (e.g. 22)
                - peak_quantile:    Quantile for peak hours      (e.g. 0.75)
                - offpeak_quantile: Quantile for off-peak hours  (e.g. 0.50)
            alpha: L2 regularisation strength passed to the IRLS solver.
            feature_weights: Optional per-feature influence weights (length =
                n_features), applied as the per-feature ridge penalty
                ``alpha / w_j**2``. Requires ``standardize=True`` and a
                non-trivial ``alpha`` to have a meaningful, comparable effect.
            standardize: When True, z-score the feature matrix on the training
                statistics and reuse those stats at predict time. This makes the
                uniform penalty act uniformly across features and is a
                prerequisite for ``feature_weights`` to behave predictably.
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

        # Fitted state: single model or per-period models.
        # Each entry is a coefficient vector from _fit_quantile_irls.
        self._coeffs: Optional[np.ndarray] = None
        self._coeffs_peak: Optional[np.ndarray] = None
        self._coeffs_offpeak: Optional[np.ndarray] = None

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

        # Fit the scaler on the FULL training matrix, then standardize once so
        # both peak/off-peak sub-models share the same scaling.
        self._fit_scaler(X_train)
        Xs = self._apply_scaler(X_train)

        if self.dynamic_config is not None and hours_train is not None:
            print("  - Training Dynamic Quantile Regression (IRLS)...")
            self._train_dynamic(Xs, y_train, hours_train)
        else:
            print(f"  - Training Quantile Regression (q={self.quantile:.2f}, IRLS)...")
            self._coeffs = _fit_quantile_irls(
                Xs, y_train, self.quantile, self.alpha,
                reg_weights=self.feature_weights,
            )

    def _train_dynamic(
        self,
        X_std: np.ndarray,
        y_train: np.ndarray,
        hours_train: np.ndarray,
    ) -> None:
        """Train separate IRLS models for peak and off-peak hours.

        ``X_std`` is already standardized by ``train`` (the scaler is fit on the
        full matrix so both sub-models and the future frame share one scaling).
        """
        peak_start = self.dynamic_config['peak_start']
        peak_end = self.dynamic_config['peak_end']
        peak_quantile = self.dynamic_config['peak_quantile']
        offpeak_quantile = self.dynamic_config['offpeak_quantile']

        peak_mask = (hours_train >= peak_start) & (hours_train <= peak_end)
        offpeak_mask = ~peak_mask

        if np.any(peak_mask):
            n_peak = int(np.sum(peak_mask))
            print(f"    - Peak hours ({peak_start}–{peak_end}): "
                  f"q={peak_quantile:.2f}, {n_peak} samples")
            self._coeffs_peak = _fit_quantile_irls(
                X_std[peak_mask], y_train[peak_mask], peak_quantile,
                self.alpha, reg_weights=self.feature_weights,
            )

        if np.any(offpeak_mask):
            n_offpeak = int(np.sum(offpeak_mask))
            print(f"    - Off-peak hours: q={offpeak_quantile:.2f}, {n_offpeak} samples")
            self._coeffs_offpeak = _fit_quantile_irls(
                X_std[offpeak_mask], y_train[offpeak_mask], offpeak_quantile,
                self.alpha, reg_weights=self.feature_weights,
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
            hours: Hour-of-day values, required when using dynamic quantile.

        Returns:
            Predicted values, shape (n_samples,).
        """
        # Standardize with the stored TRAINING stats (no-op when standardize is
        # off). Done here, on the raw matrix, so predict_iterative can keep
        # writing raw-kW predictions back into the power-lag columns without
        # any unit mismatch — the scaling is applied per call, after the lag
        # write-backs, inside this method.
        X = self._apply_scaler(np.asarray(X, dtype=np.float64))
        if self.dynamic_config is not None and hours is not None:
            return self._predict_dynamic(X, hours)
        if self._coeffs is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return _predict_quantile_irls(X, self._coeffs)

    def _predict_dynamic(self, X: np.ndarray, hours: np.ndarray) -> np.ndarray:
        """Route predictions through the appropriate peak/off-peak model."""
        peak_start = self.dynamic_config['peak_start']
        peak_end = self.dynamic_config['peak_end']

        predictions = np.zeros(len(X), dtype=np.float64)
        peak_mask = (hours >= peak_start) & (hours <= peak_end)
        offpeak_mask = ~peak_mask

        if np.any(peak_mask) and self._coeffs_peak is not None:
            predictions[peak_mask] = _predict_quantile_irls(
                X[peak_mask], self._coeffs_peak
            )

        if np.any(offpeak_mask) and self._coeffs_offpeak is not None:
            predictions[offpeak_mask] = _predict_quantile_irls(
                X[offpeak_mask], self._coeffs_offpeak
            )

        return predictions

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
) -> Dict[str, Any]:
    """
    Perform iterative (auto-regressive) prediction, feeding each predicted
    value back as the lag feature for subsequent steps.

    Simulates real-time forecasting where future actual power values are
    unknown. Falls back to a single vectorised call when no power lag
    features are present.

    Args:
        X_test:       Test feature matrix.
        y_test:       Test target vector (used for evaluation by caller).
        model:        Trained QuantileRegressionModel.
        features:     List of feature names corresponding to columns of X_test.
        n_power_lags: Number of power lag features in the model.
        hours_test:   Hour-of-day values for dynamic quantile (optional).

    Returns:
        Dict with keys:
            'predictions': np.ndarray of predicted values.
            'iterative':   bool — True if auto-regressive loop was used.
    """
    n_samples = len(X_test)
    predictions = np.zeros(n_samples, dtype=np.float64)

    # Indices of power-lag columns inside the feature matrix.
    power_lag_indices = [i for i, feat in enumerate(features) if 'power_lag' in feat]

    if not power_lag_indices:
        # No lag features — predict in one vectorised pass.
        return {
            'predictions': model.predict(X_test, hours_test),
            'iterative': False,
        }

    # Auto-regressive loop: update lag columns with previous predictions.
    for i in range(n_samples):
        X_current = X_test[i: i + 1].copy()

        if i > 0:
            for lag_order, feat_idx in enumerate(power_lag_indices, start=1):
                if i >= lag_order:
                    X_current[0, feat_idx] = predictions[i - lag_order]

        hour_current = None if hours_test is None else hours_test[i: i + 1]
        predictions[i] = model.predict(X_current, hour_current)[0]

    return {
        'predictions': predictions,
        'iterative': True,
    }
