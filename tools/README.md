# tools/

Developer utilities. Not shipped with the integration (excluded from the HACS
release zip).

## backtest.py — offline backtest + settings sweep

Tune HA Power Predictor settings without touching your live instance.

1. In Home Assistant, press **Export Training Data** or **Export All History**
   on the predictor (Settings → Devices & Services → the predictor → the button,
   or add it to a dashboard). A JSON file is written to your HA **config
   directory** (the path is shown in a notification), e.g.
   `power_predictor_export_full_20260620_153000.json`.
2. Copy that file to this repo (or anywhere on a machine with Python + numpy +
   pandas).
3. Run:

   ```bash
   # Single backtest using the exact settings from the export:
   python tools/backtest.py power_predictor_export_full_*.json

   # Grid sweep to find better settings (edit SWEEP_GRID in the script to taste):
   python tools/backtest.py power_predictor_export_full_*.json --sweep

   # Override the holdout horizon / training window:
   python tools/backtest.py EXPORT.json --horizon 48 --history-days 30
   ```

### What it does

It replays the integration's **exact** pipeline using its own pure modules
(`data_processing.py`, `models.py`), so offline results match Home Assistant. It
trains on all but the last `--horizon` hours and forecasts those hours
auto-regressively, using the *actual* temperatures for the holdout as a perfect
weather forecast (this isolates model quality from forecast quality). Metrics:

- `mae`, `rmse` — forecast error vs the held-out actuals (lower is better).
- `coverage_pct` — % of actuals at or below the forecast (calibration; a 0.75
  quantile model should land near 75%).
- `sharpness` — predicted peak-to-trough swing ÷ actual swing. **~1.0 is the
  goal**; `<1` means the forecast is too flat (the problem cyclical hour features
  address), `>1` means it overshoots.
- `corr` — correlation of the predicted vs actual shape.

### Notes

- `--sweep` runs a full cartesian product of `SWEEP_GRID` — keep the grid small.
- Use **Export All History** if you want to sweep `--history-days`, since you
  can only backtest windows shorter than what you exported.
- Requires `numpy` and `pandas` only; no Home Assistant install needed.

## cv_sweep.py — walk-forward cross-validated sweep (more robust)

`backtest.py` uses a single holdout, which on a short export can be misleading (a
config can look great on one 48 h window and diverge on the next). `cv_sweep.py`
is the robust version: it spreads several holdout windows across the whole export
(the **same** windows for every config, so they're comparable), trains each on
the `history_days` window preceding its holdout, and aggregates across folds.

```bash
python tools/cv_sweep.py power_predictor_export_full_*.json
python tools/cv_sweep.py EXPORT.json --horizon 24 --folds 10
python tools/cv_sweep.py EXPORT.json --json cv_results.json   # also dump full results
```

It ranks configs by **mean** RMSE but also reports `max_rmse` (worst fold) and
`n_diverged` (folds where the forecast swing exceeds 2× reality), so unstable
settings are flagged rather than hidden by an average — only `n_diverged = 0`
configs are trustworthy. The default model defaults were chosen with this tool.

Caveats: a config needing N days of history can't be evaluated in the first N
days of the export, so very long `history_days` is validated on a later (and
possibly seasonally narrower) slice. The perfect-weather assumption (actual
holdout temps) flatters configs that lean on temperature — under real forecast
error, temperature-light settings hold up better than the backtest suggests.
Edit `SWEEP_GRID` and the fixed `WEIGHT_TIME` / `ALPHA` / `N_TEMP_LAGS` at the
top of the script to explore other levers.
