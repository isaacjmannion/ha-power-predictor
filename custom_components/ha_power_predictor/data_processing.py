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

import numpy as np
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


def _hour_cyclical_names(hour_harmonics: int) -> list[str]:
    """Column names for the sin/cos hour-of-day harmonics, in fixed order."""
    names: list[str] = []
    for k in range(1, hour_harmonics + 1):
        names.append(f"hour_sin_{k}")
        names.append(f"hour_cos_{k}")
    return names


def add_cyclical_features(df: pd.DataFrame, hour_harmonics: int = 2) -> pd.DataFrame:
    """
    Append cyclical (sin/cos) encodings of the hour-of-day.

    A single linear ``hour`` term can only describe a monotonic ramp across the
    day, so it cannot represent the typical bimodal household load curve (a
    morning and an evening peak). Encoding hour as ``hour_sin_k`` / ``hour_cos_k``
    for k = 1..hour_harmonics gives the linear model a Fourier basis that *can*
    bend into multiple daily peaks — this is the main lever for forecast
    sharpness.

    The raw ``hour`` column is left untouched: the coordinator still passes it
    separately for peak/off-peak routing. Encoding is a pure function of
    ``hour`` so historical and future frames encode identically by construction.

    Args:
        df:             DataFrame containing an integer ``hour`` column (0–23).
        hour_harmonics: Number of harmonics (0 = no cyclical columns added).

    Returns:
        A copy of ``df`` with the cyclical columns appended, or ``df`` unchanged
        when ``hour_harmonics <= 0``.
    """
    if hour_harmonics <= 0:
        return df

    df_out = df.copy()
    hour = df_out["hour"].to_numpy(dtype=float)
    for k in range(1, hour_harmonics + 1):
        angle = 2.0 * np.pi * k * hour / 24.0
        df_out[f"hour_sin_{k}"] = np.sin(angle)
        df_out[f"hour_cos_{k}"] = np.cos(angle)
    return df_out


def get_default_features(hour_harmonics: int = 2) -> list[str]:
    """
    Return the base feature list used by the model, in column order.

    With ``hour_harmonics > 0`` the linear ``hour`` term is replaced by
    ``hour_sin_k`` / ``hour_cos_k`` cyclical columns (see ``add_cyclical_features``);
    with ``hour_harmonics == 0`` the original linear ``hour`` term is kept.

    Lag feature names (``power_lag_*``, ``temp_lag_*``) are appended dynamically
    by the coordinator based on the configured n_power_lags and n_temp_lags.
    """
    features = ["year", "month", "day_of_week"]
    if hour_harmonics > 0:
        features += _hour_cyclical_names(hour_harmonics)
    else:
        features.append("hour")
    features.append("temperature")
    return features


def build_feature_weights(
    features: list[str],
    group_weights: dict[str, float],
) -> np.ndarray:
    """
    Expand per-group influence weights onto a per-column weight vector.

    Groups:
      - ``time``        — the hour features (``hour`` or ``hour_sin_*`` / ``hour_cos_*``)
      - ``temperature`` — ``temperature`` and ``temp_lag_*``
      - ``lags``        — ``power_lag_*``
    Calendar columns (``year``, ``month``, ``day_of_week``) are always weight 1.0.

    The weights are consumed by the model as a per-feature ridge penalty
    (``reg_diag[j] = alpha / w_j**2``): a larger weight means a smaller penalty
    and therefore *more* influence for that feature group. Missing groups
    default to 1.0 (neutral). Note the weights only have a meaningful effect on
    a standardized fit with a non-trivial ``alpha`` — see the model.

    Args:
        features:      Feature names in model column order.
        group_weights: Mapping with optional keys ``time`` / ``temperature`` / ``lags``.

    Returns:
        1-D float array of length ``len(features)`` aligned to ``features``.
    """
    time_w = float(group_weights.get("time", 1.0))
    temp_w = float(group_weights.get("temperature", 1.0))
    lag_w = float(group_weights.get("lags", 1.0))

    weights: list[float] = []
    for feat in features:
        if feat == "hour" or feat.startswith("hour_sin") or feat.startswith("hour_cos"):
            weights.append(time_w)
        elif feat == "temperature" or feat.startswith("temp_lag"):
            weights.append(temp_w)
        elif feat.startswith("power_lag"):
            weights.append(lag_w)
        else:
            weights.append(1.0)
    return np.asarray(weights, dtype=float)


def _export_stat_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize recorder statistics rows for JSON export.

    Keeps only ``start`` and ``mean`` (all ``process_ha_statistics`` needs) and
    coerces ``start`` to a float Unix epoch (seconds) so it is JSON-serializable
    and round-trips through ``_parse_start``. Datetime/Timestamp starts (older
    HA) are converted via ``.timestamp()``; numeric starts pass through. ``mean``
    is kept verbatim, including ``None`` (the local pipeline drops None means just
    like production), preserving raw fidelity.
    """
    out: list[dict[str, Any]] = []
    for rec in rows:
        start = rec.get("start")
        if isinstance(start, (int, float)):
            start = float(start)
        elif hasattr(start, "timestamp"):
            start = float(start.timestamp())
        out.append({"start": start, "mean": rec.get("mean")})
    return out


def build_export_payload(
    power_stats: list[dict[str, Any]],
    temp_stats: list[dict[str, Any]],
    config: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assemble the self-describing data-export payload.

    Pure (no Home Assistant): the coordinator fetches the recorder rows and
    resolves the config, then calls this to build the JSON-serializable dict.
    The raw ``power_stats`` / ``temperature_stats`` are exported (not a processed
    frame) so an offline script can feed them straight into
    ``process_ha_statistics`` → ``add_lagged_features`` → ``add_cyclical_features``
    → the model, reproducing the live pipeline exactly.

    Args:
        power_stats: Recorder statistics rows for the power entity.
        temp_stats:  Recorder statistics rows for the temperature entity.
        config:      Resolved integration settings (entry.data + entry.options).
        meta:        Extra metadata to merge in (export time, version, entities…).
                     ``n_power_rows`` / ``n_temperature_rows`` are filled in here.

    Returns:
        ``{"meta": {...}, "config": {...}, "power_stats": [...],
           "temperature_stats": [...]}`` — all JSON-serializable.
    """
    power_rows = _export_stat_rows(power_stats)
    temp_rows = _export_stat_rows(temp_stats)

    full_meta: dict[str, Any] = dict(meta or {})
    full_meta.setdefault("schema_version", 1)
    full_meta["n_power_rows"] = len(power_rows)
    full_meta["n_temperature_rows"] = len(temp_rows)

    return {
        "meta": full_meta,
        "config": dict(config),
        "power_stats": power_rows,
        "temperature_stats": temp_rows,
    }


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
