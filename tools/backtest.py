"""
Offline backtest + settings sweep for HA Power Predictor.

Loads a JSON export produced by the integration's "Export Training Data" /
"Export All History" buttons and replays the EXACT pipeline the integration
runs in Home Assistant — process_ha_statistics -> add_lagged_features ->
add_cyclical_features -> QuantileRegressionModel -> predict_iterative — using
the integration's own pure modules, so results match production.

It does a walk-forward holdout backtest: train on all but the last `horizon`
hours, forecast those hours auto-regressively (with the *actual* temperatures as
a perfect weather forecast, to isolate model quality from forecast quality), and
compare to the held-out actuals.

Usage:
    python tools/backtest.py EXPORT.json                 # backtest with exported config
    python tools/backtest.py EXPORT.json --sweep         # grid sweep over key settings
    python tools/backtest.py EXPORT.json --horizon 48 --history-days 30

Requires numpy + pandas (the integration's runtime deps). No Home Assistant.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Make the integration's pure modules importable as top-level modules (same
# trick tests/pure/conftest.py uses), so this script reuses the real code.
_PKG = Path(__file__).resolve().parents[1] / "custom_components" / "ha_power_predictor"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import const as c  # noqa: E402
import data_processing as dp  # noqa: E402
import models as m  # noqa: E402


def _cfg(config: dict, key: str, default):
    """Read a setting from the exported config, falling back to the default."""
    value = config.get(key, default)
    return value if value is not None else default


def build_frame(power_stats, temp_stats, config, history_days=None):
    """Reproduce the integration's feature frame from raw exported stats."""
    df = dp.process_ha_statistics(power_stats, temp_stats)
    if history_days is not None:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=int(history_days))
        df = df[df["timestamp"] > cutoff].reset_index(drop=True)

    n_power_lags = int(_cfg(config, c.CONF_N_POWER_LAGS, c.DEFAULT_N_POWER_LAGS))
    n_temp_lags = int(_cfg(config, c.CONF_N_TEMP_LAGS, c.DEFAULT_N_TEMP_LAGS))
    hour_harmonics = int(_cfg(config, c.CONF_HOUR_HARMONICS, c.DEFAULT_HOUR_HARMONICS))

    df = dp.add_lagged_features(df, n_power_lags, n_temp_lags)
    df = dp.add_cyclical_features(df, hour_harmonics)

    features = dp.get_default_features(hour_harmonics)
    features += [f"power_lag_{i}" for i in range(1, n_power_lags + 1)]
    features += [f"temp_lag_{i}" for i in range(1, n_temp_lags + 1)]
    return df, features


def _metrics(actual: np.ndarray, preds: np.ndarray) -> dict:
    """MAE / RMSE / coverage / shape metrics for a forecast vs actuals."""
    actual = np.asarray(actual, dtype=float)
    preds = np.asarray(preds, dtype=float)
    actual_range = float(actual.max() - actual.min())
    pred_range = float(preds.max() - preds.min())
    corr = (
        float(np.corrcoef(actual, preds)[0, 1])
        if actual.std() > 0 and preds.std() > 0
        else float("nan")
    )
    return {
        "mae": m._mae(actual, preds),
        "rmse": m._rmse(actual, preds),
        "coverage_pct": float(np.mean(actual <= preds) * 100.0),
        # sharpness ~1.0 means the forecast's peak-to-trough swing matches reality;
        # <1 = too flat (the symptom this feature targets), >1 = overshooting.
        "sharpness": pred_range / actual_range if actual_range > 0 else float("nan"),
        "corr": corr,
    }


def _routing_hours(df, tz):
    """Local hour-of-day for peak/off-peak routing (UTC fallback if no tz),
    mirroring the integration (coordinator.py)."""
    if tz:
        return df["timestamp"].dt.tz_convert(tz).dt.hour.to_numpy()
    return df["hour"].to_numpy()


def backtest(df, features, config, horizon: int, tz=None) -> dict:
    """Train on all but the last `horizon` hours, forecast them, score them."""
    if len(df) <= horizon + c.MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Not enough data ({len(df)} rows) for horizon={horizon} "
            f"+ minimum {c.MIN_TRAINING_SAMPLES} training samples"
        )

    n_power_lags = int(_cfg(config, c.CONF_N_POWER_LAGS, c.DEFAULT_N_POWER_LAGS))
    alpha = float(_cfg(config, c.CONF_REG_ALPHA, c.DEFAULT_REG_ALPHA))
    group_weights = {
        "time": float(_cfg(config, c.CONF_WEIGHT_TIME, c.DEFAULT_WEIGHT_TIME)),
        "temperature": float(_cfg(config, c.CONF_WEIGHT_TEMPERATURE, c.DEFAULT_WEIGHT_TEMPERATURE)),
        "lags": float(_cfg(config, c.CONF_WEIGHT_LAGS, c.DEFAULT_WEIGHT_LAGS)),
    }
    feature_weights = dp.build_feature_weights(features, group_weights)
    dynamic_config = {
        "peak_start": int(_cfg(config, c.CONF_PEAK_START, c.DEFAULT_PEAK_START)),
        "peak_end": int(_cfg(config, c.CONF_PEAK_END, c.DEFAULT_PEAK_END)),
        "peak_quantile": float(_cfg(config, c.CONF_PEAK_QUANTILE, c.DEFAULT_PEAK_QUANTILE)),
        "offpeak_quantile": float(
            _cfg(config, c.CONF_OFFPEAK_QUANTILE, c.DEFAULT_OFFPEAK_QUANTILE)
        ),
    }

    train_df = df.iloc[:-horizon]
    hold_df = df.iloc[-horizon:]
    hours = _routing_hours(df, tz)            # local-hour routing, like production
    h_train, h_hold = hours[:-horizon], hours[-horizon:]

    model = m.QuantileRegressionModel(
        dynamic_config=dynamic_config,
        alpha=alpha,
        feature_weights=feature_weights,
        standardize=True,
    )
    model.train(train_df[features].values, train_df["consumption"].values, h_train)

    # Median state model: seeds the power lags so the AR forecast propagates the
    # median (not the conservative quantile) — matches the integration.
    state_model = m.QuantileRegressionModel(
        dynamic_config={**dynamic_config, "peak_quantile": 0.5, "offpeak_quantile": 0.5},
        alpha=alpha,
        feature_weights=feature_weights,
        standardize=True,
    )
    state_model.train(train_df[features].values, train_df["consumption"].values, h_train)

    # Auto-regressive forecast over the holdout. The lag columns of hold_df were
    # computed over the full series, so the first rows are seeded with real
    # recent actuals (as in production); predict_iterative then propagates the
    # state model's median through the lags.
    result = m.predict_iterative(
        hold_df[features].values,
        np.zeros(len(hold_df)),
        model,
        features,
        n_power_lags,
        hours_test=h_hold,
        state_model=state_model,
    )
    return _metrics(hold_df["consumption"].values, result["predictions"])


# Grid swept by --sweep. Edit freely; keep it small (it is a full cartesian product).
SWEEP_GRID = {
    c.CONF_HOUR_HARMONICS: [0, 1, 2, 3],
    c.CONF_REG_ALPHA: [0.1, 1.0, 3.0],
    c.CONF_WEIGHT_TIME: [1.0, 2.0],
    c.CONF_WEIGHT_TEMPERATURE: [0.5, 1.0],
}


def sweep(
    power_stats, temp_stats, base_config, horizon, history_days, grid, tz=None
) -> pd.DataFrame:
    """Backtest every combination in `grid`, ranked by RMSE (best first)."""
    keys = list(grid)
    rows = []
    for combo in product(*(grid[k] for k in keys)):
        config = dict(base_config)
        for key, value in zip(keys, combo):
            config[key] = value
        row = {key: config[key] for key in keys}
        try:
            df, features = build_frame(power_stats, temp_stats, config, history_days)
            row.update(backtest(df, features, config, horizon, tz=tz))
        except Exception as exc:  # keep sweeping if one combo fails
            row.update({"mae": float("nan"), "rmse": float("nan"),
                        "coverage_pct": float("nan"), "sharpness": float("nan"),
                        "corr": float("nan"), "error": str(exc)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse", na_position="last").reset_index(drop=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("export", help="Path to the exported JSON file")
    parser.add_argument("--sweep", action="store_true", help="Run a settings grid sweep")
    parser.add_argument("--horizon", type=int, default=48, help="Forecast horizon in hours")
    parser.add_argument(
        "--history-days", type=int, default=None,
        help="Limit training to the most recent N days (default: all exported data)",
    )
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.export).read_text(encoding="utf-8"))
    power_stats = payload["power_stats"]
    temp_stats = payload["temperature_stats"]
    base_config = dict(payload.get("config", {}))
    meta = payload.get("meta", {})
    print(
        f"Loaded {meta.get('n_power_rows')} power / {meta.get('n_temperature_rows')} "
        f"temperature rows (exported {meta.get('exported_at')}, v{meta.get('version')})"
    )
    tz = meta.get("timezone")  # peak/off-peak routing uses local time (production parity)
    print(
        f"  power source:       {meta.get('power_entity')} ({meta.get('power_entity_name')})\n"
        f"  temperature source: {meta.get('temperature_entity')} "
        f"({meta.get('temperature_entity_name')})\n"
        f"  routing timezone:   {tz or 'UTC (no tz in export)'}"
    )

    if args.sweep:
        results = sweep(
            power_stats, temp_stats, base_config, args.horizon, args.history_days, SWEEP_GRID, tz=tz
        )
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", None)
        print(f"\nSweep ({len(results)} combos, horizon={args.horizon}h), best RMSE first:\n")
        print(results.to_string(index=False))
    else:
        df, features = build_frame(power_stats, temp_stats, base_config, args.history_days)
        metrics = backtest(df, features, base_config, args.horizon, tz=tz)
        print(f"\nBacktest (exported config, horizon={args.horizon}h):")
        for key, value in metrics.items():
            print(f"  {key:>14}: {value:.4f}")


if __name__ == "__main__":
    main()
