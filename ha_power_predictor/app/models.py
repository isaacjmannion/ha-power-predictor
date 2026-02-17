"""
Quantile Regression model for power consumption prediction.
Supports both standard and dynamic peak/off-peak quantile modeling.
"""

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Optional, Any


class QuantileRegressionModel:
    """Quantile Regression model for conservative power forecasting."""
    
    def __init__(self, quantile: float = 0.75, dynamic_config: Optional[Dict] = None):
        """
        Initialize quantile regression model.
        
        Args:
            quantile: Quantile to predict (0.5-0.99)
            dynamic_config: Optional dict with peak/offpeak configuration:
                - peak_start: Hour when peak starts (e.g., 9)
                - peak_end: Hour when peak ends (e.g., 22)
                - peak_quantile: Quantile for peak hours (e.g., 0.75)
                - offpeak_quantile: Quantile for off-peak hours (e.g., 0.50)
        """
        self.quantile = quantile
        self.dynamic_config = dynamic_config
        self.model = None
        self.models_by_hour = {}
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hours_train: Optional[np.ndarray] = None
    ):
        """
        Train the quantile regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            hours_train: Hour values for dynamic quantile training
        """
        if self.dynamic_config is not None and hours_train is not None:
            print(f"  - Training Dynamic Quantile Regression...")
            self._train_dynamic(X_train, y_train, hours_train)
        else:
            print(f"  - Training Quantile Regression (q={self.quantile:.2f})...")
            self.model = QuantileRegressor(
                quantile=self.quantile,
                alpha=0.01,
                solver='highs'
            )
            self.model.fit(X_train, y_train)
            
    def _train_dynamic(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hours_train: np.ndarray
    ):
        """Train separate models for peak and off-peak hours."""
        peak_start = self.dynamic_config['peak_start']
        peak_end = self.dynamic_config['peak_end']
        peak_quantile = self.dynamic_config['peak_quantile']
        offpeak_quantile = self.dynamic_config['offpeak_quantile']
        
        # Create masks for peak and off-peak hours
        peak_mask = (hours_train >= peak_start) & (hours_train <= peak_end)
        offpeak_mask = ~peak_mask
        
        # Train peak model
        if np.any(peak_mask):
            print(f"    - Peak hours ({peak_start}-{peak_end}): q={peak_quantile:.2f}, {np.sum(peak_mask)} samples")
            self.models_by_hour['peak'] = QuantileRegressor(
                quantile=peak_quantile,
                alpha=0.01,
                solver='highs'
            )
            self.models_by_hour['peak'].fit(X_train[peak_mask], y_train[peak_mask])
        
        # Train off-peak model
        if np.any(offpeak_mask):
            print(f"    - Off-peak hours: q={offpeak_quantile:.2f}, {np.sum(offpeak_mask)} samples")
            self.models_by_hour['offpeak'] = QuantileRegressor(
                quantile=offpeak_quantile,
                alpha=0.01,
                solver='highs'
            )
            self.models_by_hour['offpeak'].fit(X_train[offpeak_mask], y_train[offpeak_mask])
    
    def predict(
        self,
        X: np.ndarray,
        hours: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make predictions using appropriate model(s).
        
        Args:
            X: Features to predict on
            hours: Hour values for dynamic quantile (required if using dynamic config)
        
        Returns:
            Array of predictions
        """
        if self.dynamic_config is not None and hours is not None:
            return self._predict_dynamic(X, hours)
        else:
            return self.model.predict(X)
    
    def _predict_dynamic(self, X: np.ndarray, hours: np.ndarray) -> np.ndarray:
        """Predict using dynamic quantile models."""
        peak_start = self.dynamic_config['peak_start']
        peak_end = self.dynamic_config['peak_end']
        
        predictions = np.zeros(len(X))
        peak_mask = (hours >= peak_start) & (hours <= peak_end)
        offpeak_mask = ~peak_mask
        
        # Predict for peak hours
        if np.any(peak_mask) and 'peak' in self.models_by_hour:
            predictions[peak_mask] = self.models_by_hour['peak'].predict(X[peak_mask])
        
        # Predict for off-peak hours
        if np.any(offpeak_mask) and 'offpeak' in self.models_by_hour:
            predictions[offpeak_mask] = self.models_by_hour['offpeak'].predict(X[offpeak_mask])
        
        return predictions
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dict with r2, mae, and rmse metrics
        """
        return {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def calculate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate percentage of actual values at or below predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Coverage percentage (0-100)
        """
        return np.mean(y_true <= y_pred) * 100


def predict_iterative(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: QuantileRegressionModel,
    features: list,
    n_power_lags: int,
    hours_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Perform iterative prediction, using predicted values for power lags.
    Simulates real-time forecasting where future actual power values aren't known.
    
    Args:
        X_test: Test features
        y_test: Test targets (for evaluation)
        model: Trained QuantileRegressionModel
        features: List of feature names
        n_power_lags: Number of power lag features
        hours_test: Hour values for dynamic quantile (optional)
    
    Returns:
        Dict with predictions and metadata
    """
    n_samples = len(X_test)
    predictions = np.zeros(n_samples)
    
    # Find indices of power lag features
    power_lag_indices = [i for i, feat in enumerate(features) if 'power_lag' in feat]
    
    # If no power lags, just predict directly
    if len(power_lag_indices) == 0:
        predictions = model.predict(X_test, hours_test)
        return {
            'predictions': predictions,
            'iterative': False
        }
    
    # Iterative prediction
    for i in range(n_samples):
        X_current = X_test[i:i + 1].copy()
        
        # Update power lag features with previous predictions
        if i > 0:
            for lag_idx, feat_idx in enumerate(power_lag_indices, start=1):
                if i >= lag_idx:
                    X_current[0, feat_idx] = predictions[i - lag_idx]
        
        # Predict
        hour_current = None if hours_test is None else hours_test[i:i + 1]
        predictions[i] = model.predict(X_current, hour_current)[0]
    
    return {
        'predictions': predictions,
        'iterative': True
    }
