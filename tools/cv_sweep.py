"""
Walk-forward cross-validated settings sweep for HA Power Predictor.

A more robust companion to backtest.py: instead of a single holdout, it spreads
`--folds` holdout windows across the whole export (the SAME windows for every
config, so configs are comparable), trains each on the `history_days` window
preceding the holdout (mirroring production), forecasts `--horizon` hours
auto-regressively with the actual temperatures as a perfect weather forecast,
and aggregates metrics across folds. It reports mean AND worst-fold RMSE plus a
divergence count (folds where the forecast swing exceeds 2x reality) so unstable
settings are surfaced, not hidden by an average.

Usage:
    python tools/cv_sweep.py EXPORT.json                       # default grid, 48h, 8 folds
    python tools/cv_sweep.py EXPORT.json --horizon 24 --folds 10
    python tools/cv_sweep.py EXPORT.json --json results.json   # also dump full results

Edit SWEEP_GRID / the fixed levers below to taste. Requires numpy + pandas; no
Home Assistant. Caveat: a config that needs N days of history cannot be
evaluated in the first N days of the export, so very long history_days values
are validated on a later (and possibly seasonally narrower) slice of the data.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parents[1] / "custom_components" / "ha_power_predictor"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import const as c  # noqa: E402
import data_processing as dp  # noqa: E402
import models as m  # noqa: E402

# Levers swept (cartesian product) — keep small; this is a full grid search.
SWEEP_GRID = {
    "history_days": [14, 30, 60],
    "hour_harmonics": [1, 2, 3],
    "n_power_lags": [3, 5, 10],
    "weight_lags": [0.5, 1.0, 2.0],
    "weight_temperature": [0.0, 1.0],
}
# Levers held fixed (edit to explore them, or move into SWEEP_GRID).
WEIGHT_TIME = 2.0
ALPHA = 0.1
N_TEMP_LAGS = 0
DIVERGENCE_SHARPNESS = 2.0  # a fold whose forecast swings > 2x reality is "diverged"


def _metrics(actual, preds):
    actual = np.asarray(actual, dtype=float)
    preds = np.asarray(preds, dtype=float)
    arange = float(actual.max() - actual.min())
    prange = float(preds.max() - preds.min())
    corr = (float(np.corrcoef(actual, preds)[0, 1])
            if actual.std() > 0 and preds.std() > 0 else float("nan"))
    return {
        "rmse": m._rmse(actual, preds),
        "cov": float(np.mean(actual <= preds) * 100.0),
        "sharp": prange / arange if arange > 0 else float("nan"),
        "corr": corr,
    }


def eval_combo(full, dyn, origins, horizon, hd, harm, npl, wlag, wtemp):
    hist = hd * 24
    feats = (dp.get_default_features(harm)
             + [f"power_lag_{i}" for i in range(1, npl + 1)]
             + [f"temp_lag_{i}" for i in range(1, N_TEMP_LAGS + 1)])
    fw = dp.build_feature_weights(feats, {"time": WEIGHT_TIME, "temperature": wtemp, "lags": wlag})
    rmses, covs, sharps, corrs = [], [], [], []
    for o in origins:
        if o - hist < 0:
            continue
        sl = full.iloc[o - hist: o + horizon].reset_index(drop=True)
        sl = dp.add_cyclical_features(dp.add_lagged_features(sl, npl, N_TEMP_LAGS), harm)
        if len(sl) <= horizon + c.MIN_TRAINING_SAMPLES:
            continue
        tr, ho = sl.iloc[:-horizon], sl.iloc[-horizon:]
        model = m.QuantileRegressionModel(
            dynamic_config=dyn, alpha=ALPHA, feature_weights=fw, standardize=True,
        )
        model.train(tr[feats].values, tr["consumption"].values, tr["hour"].values)
        res = m.predict_iterative(ho[feats].values, np.zeros(len(ho)), model, feats, npl,
                                  hours_test=ho["hour"].values)
        met = _metrics(ho["consumption"].values, res["predictions"])
        rmses.append(met["rmse"])
        covs.append(met["cov"])
        sharps.append(met["sharp"])
        corrs.append(met["corr"])
    if not rmses:
        return None
    return {
        "history_days": hd, "hour_harmonics": harm, "n_power_lags": npl,
        "weight_lags": wlag, "weight_temperature": wtemp,
        "mean_rmse": round(float(np.mean(rmses)), 3), "max_rmse": round(float(np.max(rmses)), 3),
        "mean_cov": round(float(np.mean(covs)), 1), "mean_sharp": round(float(np.mean(sharps)), 3),
        "mean_corr": round(float(np.mean(corrs)), 3),
        "n_diverged": int(sum(1 for s in sharps if s > DIVERGENCE_SHARPNESS)), "folds": len(rmses),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", help="Path to the exported JSON file")
    parser.add_argument("--horizon", type=int, default=48, help="Forecast horizon in hours")
    parser.add_argument("--folds", type=int, default=8, help="Number of walk-forward folds")
    parser.add_argument("--json", default=None, help="Optional path to dump full results as JSON")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.export).read_text(encoding="utf-8"))
    cfg = dict(payload.get("config", {}))
    full = dp.process_ha_statistics(payload["power_stats"], payload["temperature_stats"])
    n = len(full)
    dyn = {
        "peak_start": int(cfg.get(c.CONF_PEAK_START, c.DEFAULT_PEAK_START)),
        "peak_end": int(cfg.get(c.CONF_PEAK_END, c.DEFAULT_PEAK_END)),
        "peak_quantile": float(cfg.get(c.CONF_PEAK_QUANTILE, c.DEFAULT_PEAK_QUANTILE)),
        "offpeak_quantile": float(cfg.get(c.CONF_OFFPEAK_QUANTILE, c.DEFAULT_OFFPEAK_QUANTILE)),
    }
    # Fixed fold origins so every config sees the same holdout windows. lo leaves
    # room for the largest training window in the grid.
    lo = max(SWEEP_GRID["history_days"]) * 24 + max(SWEEP_GRID["n_power_lags"]) + 5
    hi = n - args.horizon - 1
    if hi <= lo:
        raise SystemExit(f"Not enough data ({n} rows) for the largest history_days + horizon.")
    origins = np.linspace(lo, hi, args.folds).astype(int)

    rows = []
    with contextlib.redirect_stdout(io.StringIO()):  # silence the model's per-fit prints
        for hd, harm, npl, wlag, wtemp in itertools.product(*SWEEP_GRID.values()):
            r = eval_combo(full, dyn, origins, args.horizon, hd, harm, npl, wlag, wtemp)
            if r:
                rows.append(r)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    meta = payload.get("meta", {})
    print(f"{meta.get('power_entity_name')} | {len(df)} configs | {args.folds} folds | "
          f"horizon {args.horizon}h | {n} rows\n")
    stable = df[df["n_diverged"] == 0].sort_values("mean_rmse")
    print("=== Top 15 by mean RMSE (stable: n_diverged=0) ===")
    print(stable.head(15).to_string(index=False))
    print(f"\nUnstable configs (n_diverged>0): {int((df['n_diverged'] > 0).sum())} of {len(df)}")

    if args.json:
        Path(args.json).write_text(json.dumps({"meta": {
            "n_rows": n, "horizon": args.horizon, "folds": args.folds,
            "weight_time": WEIGHT_TIME, "alpha": ALPHA, "n_temp_lags": N_TEMP_LAGS,
            "grid": SWEEP_GRID}, "results": rows}, indent=1), encoding="utf-8")
        print(f"Full results -> {args.json}")


if __name__ == "__main__":
    main()
