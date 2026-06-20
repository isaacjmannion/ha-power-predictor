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
