"""
Data processing for HA Power Predictor.

Operates exclusively on pre-aggregated hourly statistics from the HA recorder.
Raw history ingestion and time-bin aggregation have been removed — the recorder
already provides clean hourly means, so no binning is needed here.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

_LOGGER = logging.getLogger(__name__)


def _parse_start(start: any) -> pd.Timestamp:
    """
    Parse a recorder statistics 'start' value into a UTC-aware Timestamp.

    HA's statistics_during_period returns 'start' as a Unix epoch float
    (seconds since 1970-01-01 UTC) in recent HA versions. Older versions
    may return a datetime object directly. Both are handled here.
    """
    if isinstance(start, (int, float)):
        return pd.Timestamp(start, unit="s", tz="UTC")
    ts = pd.Timestamp(start)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def process_ha_statistics(
    power_stats: list[dict[str, Any]],
    temp_stats: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert recorder statistics into a merged, feature-enriched DataFrame.

    The recorder returns a list of dicts, each with:
      - 'start':  a UTC-aware datetime object
      - 'mean':   float (hourly mean) or None if no data in that hour

    Rows where 'mean' is None are silently dropped. The two series are then
    inner-joined on their 'start' timestamps so only hours present in both
    power and temperature records are kept.

    Temporal features (year, month, day_of_week, hour) are extracted from the
    UTC timestamps. The model learns relative time patterns so local-time
    conversion is not required here — the coordinator's future feature builder
    uses the same UTC-based extraction for consistency.

    Args:
        power_stats: Statistics rows for the power consumption entity.
        temp_stats:  Statistics rows for the temperature entity.

    Returns:
        DataFrame with columns:
            timestamp    — UTC-aware datetime (hourly)
            consumption  — mean power in the entity's native unit (kW assumed)
            temperature  — mean temperature in the entity's native unit
            year         — int
            month        — int (1–12)
            day_of_week  — int (0=Monday … 6=Sunday)
            hour         — int (0–23)

    Raises:
        ValueError if either series is empty after filtering, or if the
        inner join produces no overlapping timestamps.
    """
    t_start = time.perf_counter()
    _LOGGER.debug(
        "Processing statistics: %d power records, %d temperature records",
        len(power_stats),
        len(temp_stats),
    )

    # ── Parse power statistics ───────────────────────────────────────────────
    power_records: list[dict] = []
    for rec in power_stats:
        mean = rec.get("mean")
        start = rec.get("start")
        if mean is None or start is None:
            continue
        try:
            power_records.append({
                "timestamp": _parse_start(start),
                "consumption": float(mean),
            })
        except (TypeError, ValueError):
            continue

    if not power_records:
        raise ValueError(
            "No valid power statistics found. "
            "Ensure the power entity has 'long_term_statistics' enabled in the recorder "
            "and that data exists within the configured history window."
        )

    # ── Parse temperature statistics ─────────────────────────────────────────
    temp_records: list[dict] = []
    for rec in temp_stats:
        mean = rec.get("mean")
        start = rec.get("start")
        if mean is None or start is None:
            continue
        try:
            temp_records.append({
                "timestamp": _parse_start(start),
                "temperature": float(mean),
            })
        except (TypeError, ValueError):
            continue

    if not temp_records:
        raise ValueError(
            "No valid temperature statistics found. "
            "Ensure the temperature entity has 'long_term_statistics' enabled in the recorder."
        )

    # ── Merge on timestamp ───────────────────────────────────────────────────
    power_df = pd.DataFrame(power_records)
    temp_df = pd.DataFrame(temp_records)

    df = pd.merge(power_df, temp_df, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Power and temperature statistics have no overlapping timestamps. "
            "Verify that both entities are recording data over the same period."
        )

    # ── Temporal features ────────────────────────────────────────────────────
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour

    elapsed = time.perf_counter() - t_start
    _LOGGER.debug(
        "Processing complete: %d hourly records (%s → %s) in %.2fs",
        len(df),
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[-1],
        elapsed,
    )

    return df


def add_lagged_features(
    df: pd.DataFrame,
    n_power_lags: int = 0,
    n_temp_lags: int = 0,
) -> pd.DataFrame:
    """
    Append auto-regressive lag columns for power consumption and temperature.

    For each lag i, power_lag_i = consumption shifted back i hours, and
    temp_lag_i = temperature shifted back i hours. The first max(n_power_lags,
    n_temp_lags) rows, which would contain NaN lag values, are dropped.

    Args:
        df:           DataFrame from process_ha_statistics.
        n_power_lags: Number of hourly power lag features (0 = none).
        n_temp_lags:  Number of hourly temperature lag features (0 = none).

    Returns:
        DataFrame with lag columns appended and leading NaN rows removed.
    """
    if n_power_lags == 0 and n_temp_lags == 0:
        return df

    df_out = df.copy()

    for i in range(1, n_power_lags + 1):
        df_out[f"power_lag_{i}"] = df_out["consumption"].shift(i)

    for i in range(1, n_temp_lags + 1):
        df_out[f"temp_lag_{i}"] = df_out["temperature"].shift(i)

    max_lag = max(n_power_lags, n_temp_lags)
    n_before = len(df_out)
    df_out = df_out.iloc[max_lag:].reset_index(drop=True)

    _LOGGER.debug(
        "Lag features added: dropped %d NaN rows, %d remain",
        n_before - len(df_out),
        len(df_out),
    )

    return df_out


def get_default_features() -> list[str]:
    """
    Return the base feature list used by the model.

    Lag feature names are appended dynamically by the coordinator based on
    the configured n_power_lags and n_temp_lags values.
    """
    return ["year", "month", "day_of_week", "hour", "temperature"]


def normalize_hour_offsets(raw: Any) -> dict[int, float]:
    """
    Normalize the configured hourly offsets into a {hour: offset} mapping.

    Accepts the config ObjectSelector output — a list of {"hour": int,
    "offset": float} rows — or a plain {hour: offset} mapping. Hours are coerced
    to ints kept in 0–23 and offsets to floats; the last value wins when an hour
    repeats, and malformed or out-of-range entries are silently dropped. Returns
    an empty dict for empty/None input.
    """
    if not raw:
        return {}

    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, (list, tuple)):
        items = [
            (entry.get("hour"), entry.get("offset"))
            for entry in raw
            if isinstance(entry, dict)
        ]
    else:
        return {}

    offsets: dict[int, float] = {}
    for hour_raw, offset_raw in items:
        try:
            hour = int(hour_raw)
            offset = float(offset_raw)
        except (TypeError, ValueError):
            continue
        if 0 <= hour <= 23:
            offsets[hour] = offset  # last value wins on duplicate hours
    return offsets
