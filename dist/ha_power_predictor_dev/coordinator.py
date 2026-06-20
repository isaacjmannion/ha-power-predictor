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

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from homeassistant.components import persistent_notification
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.loader import async_get_integration
from homeassistant.util import dt as dt_util
from homeassistant.util import slugify

from .const import (
    CONF_HISTORY_DAYS,
    CONF_HOUR_HARMONICS,
    CONF_HOUR_OFFSETS,
    CONF_INTEGRATION_NAME,
    CONF_MAX_FORECAST_HOURS,
    CONF_MAX_POWER,
    CONF_MIN_POWER,
    CONF_N_POWER_LAGS,
    CONF_N_TEMP_LAGS,
    CONF_OFFPEAK_QUANTILE,
    CONF_PEAK_END,
    CONF_PEAK_QUANTILE,
    CONF_PEAK_START,
    CONF_POWER_ENTITY,
    CONF_REG_ALPHA,
    CONF_TEMPERATURE_ENTITY,
    CONF_UPDATE_INTERVAL_MINUTES,
    CONF_WEATHER_FORECAST_ENTITY,
    CONF_WEIGHT_LAGS,
    CONF_WEIGHT_TEMPERATURE,
    CONF_WEIGHT_TIME,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_HOUR_HARMONICS,
    DEFAULT_HOUR_OFFSETS,
    DEFAULT_INTEGRATION_NAME,
    DEFAULT_MAX_FORECAST_HOURS,
    DEFAULT_MAX_POWER,
    DEFAULT_MIN_POWER,
    DEFAULT_N_POWER_LAGS,
    DEFAULT_N_TEMP_LAGS,
    DEFAULT_OFFPEAK_QUANTILE,
    DEFAULT_PEAK_END,
    DEFAULT_PEAK_QUANTILE,
    DEFAULT_PEAK_START,
    DEFAULT_REG_ALPHA,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    DEFAULT_WEIGHT_LAGS,
    DEFAULT_WEIGHT_TEMPERATURE,
    DEFAULT_WEIGHT_TIME,
    DOMAIN,
    EXPORT_SCHEMA_VERSION,
    MIN_TRAINING_SAMPLES,
)
from .data_processing import (
    add_cyclical_features,
    add_lagged_features,
    build_export_payload,
    build_feature_weights,
    get_default_features,
    normalize_hour_offsets,
    process_ha_statistics,
)
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
        interval_minutes = int(
            cfg.get(CONF_UPDATE_INTERVAL_MINUTES, DEFAULT_UPDATE_INTERVAL_MINUTES)
        )

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
        max_forecast_hours: int = int(cfg.get(CONF_MAX_FORECAST_HOURS, DEFAULT_MAX_FORECAST_HOURS))
        hour_offsets = normalize_hour_offsets(cfg.get(CONF_HOUR_OFFSETS, DEFAULT_HOUR_OFFSETS))
        hour_harmonics: int = int(cfg.get(CONF_HOUR_HARMONICS, DEFAULT_HOUR_HARMONICS))
        reg_alpha: float = float(cfg.get(CONF_REG_ALPHA, DEFAULT_REG_ALPHA))
        group_weights = {
            "time": float(cfg.get(CONF_WEIGHT_TIME, DEFAULT_WEIGHT_TIME)),
            "temperature": float(cfg.get(CONF_WEIGHT_TEMPERATURE, DEFAULT_WEIGHT_TEMPERATURE)),
            "lags": float(cfg.get(CONF_WEIGHT_LAGS, DEFAULT_WEIGHT_LAGS)),
        }

        min_power: float = float(cfg.get(CONF_MIN_POWER, DEFAULT_MIN_POWER))
        max_power: float = float(cfg.get(CONF_MAX_POWER, DEFAULT_MAX_POWER))
        dynamic_config = {
            "peak_start": int(cfg.get(CONF_PEAK_START, DEFAULT_PEAK_START)),
            "peak_end": int(cfg.get(CONF_PEAK_END, DEFAULT_PEAK_END)),
            "peak_quantile": float(cfg.get(CONF_PEAK_QUANTILE, DEFAULT_PEAK_QUANTILE)),
            "offpeak_quantile": float(cfg.get(CONF_OFFPEAK_QUANTILE, DEFAULT_OFFPEAK_QUANTILE)),
        }

        # ── Step 1: Fetch recorder statistics ────────────────────────────────
        _LOGGER.debug(
            "Fetching %d days of statistics for %s and %s",
            history_days,
            power_entity,
            temp_entity,
        )
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

        # ── Step 4: Add lag + cyclical hour features ─────────────────────────
        df = await self.hass.async_add_executor_job(
            add_lagged_features, df, n_power_lags, n_temp_lags
        )
        df = await self.hass.async_add_executor_job(
            add_cyclical_features, df, hour_harmonics
        )

        features: list[str] = get_default_features(hour_harmonics)
        for i in range(1, n_power_lags + 1):
            features.append(f"power_lag_{i}")
        for i in range(1, n_temp_lags + 1):
            features.append(f"temp_lag_{i}")

        # Per-feature influence weights (time / temperature / lags), expanded
        # onto the exact feature column order. Applied as a per-feature ridge
        # penalty on the standardized fit.
        feature_weights = build_feature_weights(features, group_weights)

        # ── Step 5: Train on full dataset ────────────────────────────────────
        _LOGGER.debug("Training quantile regression model on %d samples", len(df))
        model = QuantileRegressionModel(
            dynamic_config=dynamic_config,
            alpha=reg_alpha,
            feature_weights=feature_weights,
            standardize=True,
        )
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
        _LOGGER.debug("Building future feature matrix (%d hours)", max_forecast_hours)
        df_future: pd.DataFrame = await self.hass.async_add_executor_job(
            _build_future_df,
            df,
            forecast_data,
            mean_temp_fallback,
            max_forecast_hours,
            features,
            n_power_lags,
            n_temp_lags,
            hour_harmonics,
        )

        # ── Step 7: Generate predictions ─────────────────────────────────────
        _LOGGER.debug("Generating %d-hour iterative predictions", max_forecast_hours)
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

        predictions: list[dict[str, Any]] = []
        for i, (_, row) in enumerate(df_future.iterrows()):
            # Add the configured offset before clamping so min/max_power still
            # bound the published value. Match on the LOCAL hour-of-day so an
            # offset for e.g. hour 13 lands at 1 pm on the user's clock — the same
            # local time the forecast is displayed in (sensor.py shows local time).
            local_hour = dt_util.as_local(row["timestamp"]).hour
            offset = hour_offsets.get(local_hour, 0.0)
            value = min(max_power, max(min_power, raw_preds[i] + offset))
            predictions.append(
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "predicted": round(float(value), 3),
                }
            )

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
            "max_forecast_hours": max_forecast_hours,
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
        state = self.hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            _LOGGER.debug(
                "Weather entity '%s' not ready (state=%s); will retry next cycle",
                entity_id,
                state.state if state else "missing",
            )
            return []
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

    def _entity_friendly_name(self, entity_id: str | None) -> str | None:
        """Return an entity's friendly (display) name, or None if unavailable."""
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        return state.name if state is not None else None

    async def async_export_data(self, days: int, label: str) -> str:
        """
        Export raw recorder statistics + the resolved config to a JSON file in
        the HA config directory, for offline analysis / backtesting.

        The raw stats are exported (not a processed frame) so an offline script
        can replay the exact pure pipeline (process → lag → cyclical → model).

        Args:
            days:  How many days back to fetch (e.g. history_days, or a larger
                   "all available" window).
            label: Short tag for the filename ("training" / "full").

        Returns:
            The absolute path of the written JSON file.
        """
        cfg = {**self.entry.data, **self.entry.options}
        power_entity: str = cfg[CONF_POWER_ENTITY]
        temp_entity: str = cfg[CONF_TEMPERATURE_ENTITY]
        weather_entity: str | None = cfg.get(CONF_WEATHER_FORECAST_ENTITY)
        integration_name: str = cfg.get(CONF_INTEGRATION_NAME, DEFAULT_INTEGRATION_NAME)

        now_utc = dt_util.utcnow()
        start_utc = now_utc - timedelta(days=days)

        power_stats, temp_stats = await get_instance(self.hass).async_add_executor_job(
            self._fetch_statistics, power_entity, temp_entity, start_utc, now_utc
        )

        try:
            integration = await async_get_integration(self.hass, DOMAIN)
            version = str(integration.version)
        except Exception:  # version is best-effort metadata only
            version = "unknown"

        meta = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "exported_at": dt_util.now().isoformat(),
            "integration": DOMAIN,
            "integration_name": integration_name,
            "version": version,
            "power_entity": power_entity,
            "power_entity_name": self._entity_friendly_name(power_entity),
            "temperature_entity": temp_entity,
            "temperature_entity_name": self._entity_friendly_name(temp_entity),
            "weather_entity": weather_entity,
            "weather_entity_name": self._entity_friendly_name(weather_entity),
            "requested_days": days,
            "scope": label,
            "timezone": str(dt_util.get_default_time_zone()),
        }
        payload = build_export_payload(power_stats, temp_stats, cfg, meta)

        timestamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{slugify(integration_name)}_export_{label}_{timestamp}.json"
        path = self.hass.config.path(filename)

        await self.hass.async_add_executor_job(_write_json_file, path, payload)

        _LOGGER.info(
            "Exported %d power + %d temperature rows (%s, %d days) to %s",
            payload["meta"]["n_power_rows"],
            payload["meta"]["n_temperature_rows"],
            label,
            days,
            path,
        )
        persistent_notification.async_create(
            self.hass,
            (
                f"Exported **{payload['meta']['n_power_rows']}** power and "
                f"**{payload['meta']['n_temperature_rows']}** temperature rows "
                f"({days} days) to:\n\n`{path}`"
            ),
            title=f"{integration_name}: data export complete",
            notification_id=f"{self.entry.entry_id}_export_{label}",
        )
        return path


def _write_json_file(path: str, payload: dict) -> None:
    """Write the export payload as pretty JSON (runs in the executor)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


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
    hour_harmonics: int = 0,
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
                # Naive datetimes are assumed to be in HA's local timezone (not UTC)
                # This handles integrations like BoM that provide timezone-naive forecasts
                ha_tz = dt_util.get_default_time_zone()
                # Resolve DST transitions explicitly so localization never raises:
                #   ambiguous=False           -> pick standard (non-DST) time on the
                #                                repeated fall-back hour
                #   nonexistent="shift_forward" -> shift a spring-forward gap time to the
                #                                next valid instant
                # Without these, the repeated/skipped hour raises AmbiguousTimeError /
                # NonExistentTimeError, which are NOT caught by the except below and would
                # crash the whole update at the DST changeover.
                dt = dt.tz_localize(
                    ha_tz, ambiguous=False, nonexistent="shift_forward"
                )
            # Convert to the working timezone (derived from historical data)
            dt = dt.tz_convert(tz) if tz else dt
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

    # Seed lag columns by concatenating the historical tail with the future rows
    # and computing .shift() on the combined series.  This ensures that:
    #   - Row 0 of future inherits its lag values from real historical data.
    #   - Row i inherits lag values from the correct preceding future row(s),
    #     so temperature lags track the forecast temperature as it evolves hour
    #     by hour rather than being frozen at a single stale value.
    #
    # For power lags, predict_iterative will overwrite these values step-by-step
    # with its own predictions as it walks forward through the forecast window,
    # but the correct historical seed here ensures the very first step is sound.
    # For temperature lags, the values computed here are used as-is — forecast
    # temperatures are already present in df_future["temperature"], so shifting
    # over the combined series gives the correct lagged temperature for every
    # future row without any further updates needed at inference time.

    max_lag = max(n_power_lags, n_temp_lags, 1)

    if n_power_lags > 0:
        hist_power_tail = df_historical["consumption"].iloc[-max_lag:].reset_index(drop=True)
        future_power = df_future["consumption"].reset_index(drop=True)
        combined_power = pd.concat([hist_power_tail, future_power], ignore_index=True)
        for i in range(1, n_power_lags + 1):
            lagged = combined_power.shift(i)
            df_future[f"power_lag_{i}"] = lagged.iloc[max_lag:].values

    if n_temp_lags > 0:
        hist_temp_tail = df_historical["temperature"].iloc[-max_lag:].reset_index(drop=True)
        future_temp = df_future["temperature"].reset_index(drop=True)
        combined_temp = pd.concat([hist_temp_tail, future_temp], ignore_index=True)
        for i in range(1, n_temp_lags + 1):
            lagged = combined_temp.shift(i)
            df_future[f"temp_lag_{i}"] = lagged.iloc[max_lag:].values

    # Cyclical hour encoding — identical transform to the training frame, so the
    # future feature columns line up with what the model was trained on.
    df_future = add_cyclical_features(df_future, hour_harmonics)

    return df_future
