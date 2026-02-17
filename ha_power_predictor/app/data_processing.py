"""
Data processing functions for Home Assistant power data.
Handles parsing from HA API, binning, and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _log(msg: str):
    """Log to both logger and stdout so it appears in the HA add-on log."""
    logger.info(msg)
    print(msg, flush=True)


def process_ha_data(
    power_data: List[Dict[str, Any]],
    temp_data: List[Dict[str, Any]],
    bin_size_minutes: int = 60,
    timezone: str = 'UTC'
) -> pd.DataFrame:
    """
    Process Home Assistant API data into binned DataFrame.
    """
    t_start = time.time()
    _log(f"📊 Processing HA data with {bin_size_minutes}-minute bins...")
    _log(f"   Input: {len(power_data)} power records, {len(temp_data)} temp records")

    # --- Parse power records ---
    _log("   [1/6] Parsing power records...")
    t = time.time()
    power_records = []
    skipped = 0
    for record in power_data:
        try:
            timestamp = pd.to_datetime(record.get('last_changed') or record.get('last_updated'))
            state = record.get('state')
            if state not in ['unknown', 'unavailable', None]:
                power_records.append({
                    'timestamp': timestamp,
                    'consumption': float(state)
                })
            else:
                skipped += 1
        except (ValueError, TypeError):
            skipped += 1
            continue

    if not power_records:
        raise ValueError("No valid power records found")

    _log(f"   [1/6] Done — {len(power_records)} valid, {skipped} skipped ({time.time()-t:.1f}s)")

    # --- Parse temperature records ---
    _log("   [2/6] Parsing temperature records...")
    t = time.time()
    temp_records = []
    skipped = 0
    for record in temp_data:
        try:
            timestamp = pd.to_datetime(record.get('last_changed') or record.get('last_updated'))
            state = record.get('state')
            if state not in ['unknown', 'unavailable', None]:
                temp_records.append({
                    'timestamp': timestamp,
                    'temperature': float(state)
                })
            else:
                skipped += 1
        except (ValueError, TypeError):
            skipped += 1
            continue

    if not temp_records:
        raise ValueError("No valid temperature records found")

    _log(f"   [2/6] Done — {len(temp_records)} valid, {skipped} skipped ({time.time()-t:.1f}s)")

    # --- Build DataFrames ---
    _log("   [3/6] Building DataFrames...")
    t = time.time()
    power_df = pd.DataFrame(power_records)
    temp_df = pd.DataFrame(temp_records)
    _log(f"   [3/6] Done ({time.time()-t:.1f}s)")

    # --- Localise timestamps ---
    _log(f"   [4/6] Localising timestamps to {timezone}...")
    t = time.time()
    tz = pytz.timezone(timezone)
    power_df['timestamp'] = power_df['timestamp'].dt.tz_convert(tz)
    temp_df['timestamp'] = temp_df['timestamp'].dt.tz_convert(tz)
    _log(f"   [4/6] Done ({time.time()-t:.1f}s)")

    # --- Bin into time slots ---
    _log(f"   [5/6] Binning into {bin_size_minutes}-minute slots...")
    t = time.time()
    bin_freq = f'{bin_size_minutes}min'

    power_df['time_bin'] = power_df['timestamp'].dt.floor(bin_freq)
    power_binned = power_df.groupby('time_bin')['consumption'].mean().reset_index()
    _log(f"         Power binned: {len(power_binned)} bins")

    temp_df['time_bin'] = temp_df['timestamp'].dt.floor(bin_freq)
    temp_binned = temp_df.groupby('time_bin')['temperature'].mean().reset_index()
    _log(f"         Temp binned:  {len(temp_binned)} bins")

    _log(f"   [5/6] Done ({time.time()-t:.1f}s)")

    # --- Merge and add features ---
    _log("   [6/6] Merging datasets and adding temporal features...")
    t = time.time()
    df_merged = pd.merge(power_binned, temp_binned, on='time_bin', how='inner')
    df_merged = df_merged.rename(columns={'time_bin': 'timestamp'})
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

    df_merged['year'] = df_merged['timestamp'].dt.year
    df_merged['month'] = df_merged['timestamp'].dt.month
    df_merged['day_of_week'] = df_merged['timestamp'].dt.dayofweek
    df_merged['hour'] = df_merged['timestamp'].dt.hour
    df_merged['minute'] = df_merged['timestamp'].dt.minute
    _log(f"   [6/6] Done ({time.time()-t:.1f}s)")

    total = time.time() - t_start
    _log(f"✅ Processing complete: {len(df_merged)} bins — "
         f"{df_merged['timestamp'].min()} → {df_merged['timestamp'].max()} "
         f"(total {total:.1f}s)")

    return df_merged


def add_lagged_features(
    df: pd.DataFrame,
    n_power_lags: int = 0,
    n_temp_lags: int = 0
) -> pd.DataFrame:
    """
    Add lagged features for previous power and temperature readings.
    """
    if n_power_lags == 0 and n_temp_lags == 0:
        return df

    _log(f"   Adding lags: {n_power_lags} power lag(s), {n_temp_lags} temp lag(s)...")
    t = time.time()
    df_lagged = df.copy()

    for i in range(1, n_power_lags + 1):
        df_lagged[f'power_lag_{i}'] = df_lagged['consumption'].shift(i)

    for i in range(1, n_temp_lags + 1):
        df_lagged[f'temp_lag_{i}'] = df_lagged['temperature'].shift(i)

    max_lag = max(n_power_lags, n_temp_lags)
    if max_lag > 0:
        before = len(df_lagged)
        df_lagged = df_lagged.iloc[max_lag:].reset_index(drop=True)
        _log(f"   Dropped {before - len(df_lagged)} rows with NaN lags, {len(df_lagged)} remain ({time.time()-t:.1f}s)")

    return df_lagged


def get_default_features() -> List[str]:
    """Get the default feature set for training."""
    return ['year', 'month', 'day_of_week', 'hour', 'temperature']
