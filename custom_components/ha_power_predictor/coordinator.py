"""
DataUpdateCoordinator for HA Power Predictor.

Runs the complete pipeline on a configurable schedule and on manual
"Train Now" button presses:

  1. Fetch hourly statistics for power + temperature from the recorder.
  2. Fetch the hourly weather forecast via a HA service call.
  3. Process statistics into a training DataFrame.
  4. Train the IRLS quantile regression model on the full dataset.
  5. Build a 48-hour future feature matrix (with forecast temperatures,
     falling back to the historical mean where the forecast is short).
  6. Generate 48-hour predictions via iterative (auto-regressive) inference.
  7. Return the prediction list so sensor entities can read it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    CONF_HISTORY_DAYS,
    CONF_MAX_POWER,
    CONF_MIN_POWER,
    CONF_N_POWER_LAGS,
    CONF_N_TEMP_LAGS,
    CONF_OFFPEAK_QUANTILE,
    CONF_PEAK_END,
    CONF_PEAK_QUANTILE,
    CONF_PEAK_START,
    CONF_POWER_ENTITY,
    CONF_QUANTILE,
    CONF_TEMPERATURE_ENTITY,
    CONF_UPDATE_INTERVAL_MINUTES,
    CONF_USE_DYNAMIC_QUANTILE,
    CONF_WEATHER_FORECAST_ENTITY,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_MAX_POWER,
    DEFAULT_MIN_POWER,
    DEFAULT_N_POWER_LAGS,
    DEFAULT_N_TEMP_LAGS,
    DEFAULT_OFFPEAK_QUANTILE,
    DEFAULT_PEAK_END,
    DEFAULT_PEAK_QUANTILE,
    DEFAULT_PEAK_START,
    DEFAULT_QUANTILE,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    DEFAULT_USE_DYNAMIC_QUANTILE,
    DOMAIN,
    MIN_TRAINING_SAMPLES,
    PREDICTION_HOURS,
)
from .data_processing import add_lagged_features, get_default_features, process_ha_statistics
from .models import QuantileRegressionModel, predict_iterative

_LOGGER = logging.getLogger(__name__)


class PowerPredictorCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """
    Coordinator that drives the full power prediction pipeline.

    Reads config from entry.data (set at setup) and entry.options (set via
    the options flow). Options always take precedence, allowing the user to
    tune parameters without removing and re-adding the integration.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.entry = entry
        cfg = {**entry.data, **entry.options}
        interval_minutes = int(cfg.get(CONF_UPDATE_INTERVAL_MINUTES, DEFAULT_UPDATE_INTERVAL_MINUTES))

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=interval_minutes),
        )

    # ------------------------------------------------------------------
    # Core update method — called by HA on schedule and by async_request_refresh
    # ------------------------------------------------------------------

    async def _async_update_data(self) -> dict[str, Any]:
        """Run the full fetch → train → predict pipeline."""

        # Merge data + options so options always win
        cfg = {**self.entry.data, **self.entry.options}

        power_entity: str = cfg[CONF_POWER_ENTITY]
        temp_entity: str = cfg[CONF_TEMPERATURE_ENTITY]
        weather_entity: str = cfg[CONF_WEATHER_FORECAST_ENTITY]
        history_days: int = int(cfg.get(CONF_HISTORY_DAYS, DEFAULT_HISTORY_DAYS))
        n_power_lags: int = int(cfg.get(CONF_N_POWER_LAGS, DEFAULT_N_POWER_LAGS))
        n_temp_lags: int = int(cfg.get(CONF_N_TEMP_LAGS, DEFAULT_N_TEMP_LAGS))
        quantile: float = float(cfg.get(CONF_QUANTILE, DEFAULT_QUANTILE))
        use_dynamic: bool = bool(cfg.get(CONF_USE_DYNAMIC_QUANTILE, DEFAULT_USE_DYNAMIC_QUANTILE))

        min_power: float = float(cfg.get(CONF_MIN_POWER, DEFAULT_MIN_POWER))
        max_power: float = float(cfg.get(CONF_MAX_POWER, DEFAULT_MAX_POWER))
        if use_dynamic:
            dynamic_config = {
                "peak_start": int(cfg.get(CONF_PEAK_START, DEFAULT_PEAK_START)),
                "peak_end": int(cfg.get(CONF_PEAK_END, DEFAULT_PEAK_END)),
                "peak_quantile": float(cfg.get(CONF_PEAK_QUANTILE, DEFAULT_PEAK_QUANTILE)),
                "offpeak_quantile": float(cfg.get(CONF_OFFPEAK_QUANTILE, DEFAULT_OFFPEAK_QUANTILE)),
            }

        # ── Step 1: Fetch recorder statistics ────────────────────────────────
        _LOGGER.debug("Fetching %d days of statistics for %s and %s", history_days, power_entity, temp_entity)
        now_utc = dt_util.utcnow()
        start_utc = now_utc - timedelta(days=history_days)

        try:
            power_stats, temp_stats = await get_instance(self.hass).async_add_executor_job(
                self._fetch_statistics,
                power_entity,
                temp_entity,
                start_utc,
                now_utc,
            )
        except Exception as err:
            raise UpdateFailed(f"Failed to fetch recorder statistics: {err}") from err

        if not power_stats:
            raise UpdateFailed(
                f"No hourly statistics found for '{power_entity}'. "
                "Check that the entity has 'long_term_statistics' enabled and "
                "that history_days does not exceed the recorder retention period."
            )
        if not temp_stats:
            raise UpdateFailed(
                f"No hourly statistics found for '{temp_entity}'. "
                "Check that the entity has 'long_term_statistics' enabled."
            )

        # ── Step 2: Fetch weather forecast ───────────────────────────────────
        forecast_data: list[dict] = await self._async_fetch_forecast(weather_entity)

        # ── Step 3: Process statistics → DataFrame ───────────────────────────
        try:
            df: pd.DataFrame = await self.hass.async_add_executor_job(
                process_ha_statistics,
                power_stats,
                temp_stats,
            )
        except ValueError as err:
            raise UpdateFailed(f"Statistics processing failed: {err}") from err

        if len(df) < MIN_TRAINING_SAMPLES:
            raise UpdateFailed(
                f"Insufficient training data: {len(df)} hourly records "
                f"(minimum {MIN_TRAINING_SAMPLES}). "
                f"Try increasing history_days or confirming both entities have data."
            )

        mean_temp_fallback = float(df["temperature"].mean())

        # ── Step 4: Add lag features ─────────────────────────────────────────
        df = await self.hass.async_add_executor_job(
            add_lagged_features, df, n_power_lags, n_temp_lags
        )

        features: list[str] = get_default_features()
        for i in range(1, n_power_lags + 1):
            features.append(f"power_lag_{i}")
        for i in range(1, n_temp_lags + 1):
            features.append(f"temp_lag_{i}")

        # ── Step 5: Train on full dataset ────────────────────────────────────
        _LOGGER.debug("Training quantile regression model on %d samples", len(df))
        model = QuantileRegressionModel(quantile=quantile, dynamic_config=dynamic_config)
        await self.hass.async_add_executor_job(
            model.train,
            df[features].values,
            df["consumption"].values,
            df["hour"].values,
        )

        # ── Step 5b: In-sample fitted values ────────────────────────────────
        # Run predict on the training data so the user can overlay the model
        # against their actual historical consumption in Lovelace charts.
        fitted_raw: np.ndarray = await self.hass.async_add_executor_job(
            model.predict,
            df[features].values,
            df["hour"].values,
        )
        fitted_coverage: float = float(
            np.mean(df["consumption"].values <= fitted_raw) * 100
        )
        # Limit to the most recent 48 hours to keep attribute size manageable
        df_fitted = df.iloc[-48:]
        fitted_raw_48 = fitted_raw[-48:]
        fitted: list[dict[str, Any]] = [
            {
                "timestamp": row["timestamp"].isoformat(),
                "value": round(float(min(max_power, max(min_power, fitted_raw_48[i]))), 3),
            }
            for i, (_, row) in enumerate(df_fitted.iterrows())
        ]
        _LOGGER.debug(
            "In-sample fit: %d records, coverage %.1f%%", len(fitted), fitted_coverage
        )

        # ── Step 6: Build future feature matrix ──────────────────────────────
        _LOGGER.debug("Building future feature matrix (%d hours)", PREDICTION_HOURS)
        df_future: pd.DataFrame = await self.hass.async_add_executor_job(
            _build_future_df,
            df,
            forecast_data,
            mean_temp_fallback,
            PREDICTION_HOURS,
            features,
            n_power_lags,
            n_temp_lags,
        )

        # ── Step 7: Generate predictions ─────────────────────────────────────
        _LOGGER.debug("Generating %d-hour iterative predictions", PREDICTION_HOURS)
        future_result = await self.hass.async_add_executor_job(
            predict_iterative,
            df_future[features].values,
            np.zeros(len(df_future)),
            model,
            features,
            n_power_lags,
            df_future["hour"].values,
        )

        raw_preds: np.ndarray = future_result["predictions"]

        predictions: list[dict[str, Any]] = [
            {
                "timestamp": row["timestamp"].isoformat(),
                "predicted": round(float(min(max_power, max(min_power, raw_preds[i]))), 3),
            }
            for i, (_, row) in enumerate(df_future.iterrows())
        ]

        _LOGGER.info(
            "Pipeline complete — %d predictions from %s to %s  (trained on %d samples)",
            len(predictions),
            predictions[0]["timestamp"],
            predictions[-1]["timestamp"],
            len(df),
        )

        return {
            "predictions": predictions,
            "fitted": fitted,
            "fitted_coverage": fitted_coverage,
            "power_entity": power_entity,
            "history_days": history_days,
            "last_updated": dt_util.now().isoformat(),
            "training_samples": len(df),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_statistics(
        self,
        power_entity: str,
        temp_entity: str,
        start: datetime,
        end: datetime,
    ) -> tuple[list[dict], list[dict]]:
        """
        Fetch hourly recorder statistics for both entities in a single call.

        Runs synchronously in the executor thread pool.
        Returns (power_stats, temp_stats) as plain lists of dicts.
        """
        raw: dict = statistics_during_period(
            self.hass,
            start,
            end,
            {power_entity, temp_entity},
            "hour",
            None,       # units — use native units stored in recorder
            {"mean"},   # we only need the hourly mean
        )
        return raw.get(power_entity, []), raw.get(temp_entity, [])

    async def _async_fetch_forecast(self, entity_id: str) -> list[dict]:
        """
        Fetch the hourly weather forecast via HA's weather.get_forecasts service.

        Returns an empty list (gracefully) if the service call fails or the
        entity does not provide a forecast — the pipeline will fall back to
        the historical mean temperature for any missing hours.
        """
        try:
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": entity_id, "type": "hourly"},
                blocking=True,
                return_response=True,
            )
            if isinstance(response, dict):
                # HA 2024.2+ wraps the result under 'service_response'
                service_data = response.get("service_response", response)
                entity_data = service_data.get(entity_id, {})
                forecasts: list[dict] = entity_data.get("forecast", [])
                _LOGGER.debug("Fetched %d forecast hours from %s", len(forecasts), entity_id)
                return forecasts
        except Exception as err:
            _LOGGER.warning(
                "Could not fetch weather forecast from '%s': %s. "
                "Predictions will use mean historical temperature for missing hours.",
                entity_id,
                err,
            )
        return []


# ---------------------------------------------------------------------------
# Future feature builder — pure function, safe to run in executor
# ---------------------------------------------------------------------------

def _build_future_df(
    df_historical: pd.DataFrame,
    forecast_data: list[dict],
    mean_temp_fallback: float,
    n_hours: int,
    features: list[str],
    n_power_lags: int,
    n_temp_lags: int,
) -> pd.DataFrame:
    """
    Build a feature DataFrame covering the next n_hours starting at the next
    full UTC hour.

    Strategy for lag features
    -------------------------
    Rather than manually seeding lag columns, we concatenate df_historical with
    the future stub rows and compute .shift() on the combined series. This means
    the first few future rows naturally inherit correct lag values from real
    historical data. predict_iterative then overwrites the power lag columns as
    it steps through the future rows, propagating its own predictions.

    Missing forecast temperatures
    ------------------------------
    Any future hour without a matching forecast entry is filled with
    mean_temp_fallback (the mean of all historical temperature values).
    """
    # Derive timezone from the historical timestamps (they carry recorder UTC tz)
    sample_ts = df_historical["timestamp"].iloc[-1]
    tz = sample_ts.tzinfo

    now = dt_util.utcnow().astimezone(tz) if tz else dt_util.utcnow()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # Build forecast temperature lookup keyed by hour-truncated aware datetimes
    forecast_lookup: dict = {}
    for fc in forecast_data:
        try:
            dt = pd.to_datetime(fc["datetime"])
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC") if tz is None else dt.tz_localize(tz)
            else:
                dt = dt.tz_convert(tz)
            dt = dt.replace(minute=0, second=0, microsecond=0)
            forecast_lookup[dt] = float(fc["temperature"])
        except (KeyError, ValueError, TypeError, AttributeError):
            continue

    _LOGGER.debug(
        "Forecast lookup built: %d hours available, fallback temp=%.1f°",
        len(forecast_lookup),
        mean_temp_fallback,
    )

    # Build future rows — consumption is 0.0 as a placeholder (overwritten by predictor)
    future_rows = []
    filled_from_forecast = 0
    filled_from_fallback = 0
    for i in range(n_hours):
        ts = next_hour + timedelta(hours=i)
        if ts in forecast_lookup:
            temp = forecast_lookup[ts]
            filled_from_forecast += 1
        else:
            temp = mean_temp_fallback
            filled_from_fallback += 1
        future_rows.append({
            "timestamp": ts,
            "temperature": temp,
            "consumption": 0.0,
            "year": ts.year,
            "month": ts.month,
            "day_of_week": ts.weekday(),
            "hour": ts.hour,
        })

    if filled_from_fallback:
        _LOGGER.debug(
            "%d of %d future hours used mean-temperature fallback (%.1f°)",
            filled_from_fallback,
            n_hours,
            mean_temp_fallback,
        )

    df_future = pd.DataFrame(future_rows)

    # Seed lag columns from the tail of historical data.
    # We need up to max(n_power_lags, n_temp_lags) rows from the end of history
    # to correctly seed the initial lag values for the first future rows.
    # predict_iterative will then overwrite power_lag columns as it steps forward,
    # propagating its own predictions — so we only need the historical seed to be
    # correct for the very first prediction step.
    if n_power_lags > 0:
        # Last n_power_lags consumption values, most-recent-first
        hist_power = df_historical["consumption"].iloc[-n_power_lags:].values[::-1]
        for i in range(1, n_power_lags + 1):
            seed_val = float(hist_power[i - 1]) if i <= len(hist_power) else float(df_historical["consumption"].mean())
            df_future[f"power_lag_{i}"] = seed_val

    if n_temp_lags > 0:
        hist_temp = df_historical["temperature"].iloc[-n_temp_lags:].values[::-1]
        for i in range(1, n_temp_lags + 1):
            seed_val = float(hist_temp[i - 1]) if i <= len(hist_temp) else float(df_historical["temperature"].mean())
            df_future[f"temp_lag_{i}"] = seed_val

    return df_future
