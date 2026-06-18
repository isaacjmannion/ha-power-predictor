# CLAUDE.md — `ha_power_predictor` package internals

Module-level guidance for the integration code. For the project overview,
validation steps, and release process, see the repo-root `CLAUDE.md`.

## Module responsibilities

| File | Responsibility |
|------|----------------|
| `__init__.py` | `async_setup_entry` / `async_unload_entry`. Creates the coordinator, runs the first refresh (`async_config_entry_first_refresh`, which raises `ConfigEntryNotReady` to make HA retry), forwards the `sensor` + `button` platforms, and registers an options-update listener that **reloads the entry** so option changes take effect immediately. |
| `coordinator.py` | `PowerPredictorCoordinator` (subclass of `DataUpdateCoordinator`). Owns `_async_update_data` — the full fetch→train→predict pipeline — plus the module-level pure function `_build_future_df`. |
| `models.py` | The ML core. `QuantileRegressionModel` + the standalone `predict_iterative`. Pure numpy — no sklearn/scipy. |
| `data_processing.py` | `process_ha_statistics`, `add_lagged_features`, `get_default_features`, `_parse_start`. Turns recorder rows into the training DataFrame. |
| `config_flow.py` | `PowerPredictorConfigFlow` (2 steps) + `PowerPredictorOptionsFlow`. Schema builders + `_coerce_numbers`. |
| `sensor.py` | Four entities: two `PowerPredictionSensor` (24h, 48h), one `ExtendedForecastSensor`, one `FittedModelSensor`. |
| `button.py` | `TrainNowButton` → calls `coordinator.async_request_refresh()`. |
| `const.py` | `DOMAIN`, `CONF_*` keys, `DEFAULT_*` values, and pipeline limits (`MIN_TRAINING_SAMPLES = 24`, `MAX_FORECAST_HOURS_LIMIT = 168`). |

## Data contracts

These shapes are passed between modules — keep them in lockstep when editing.

### Training DataFrame (from `process_ha_statistics`)

Columns: `timestamp` (UTC-aware), `consumption` (float, kW assumed),
`temperature` (float), `year`, `month`, `day_of_week` (0=Mon … 6=Sun), `hour`
(0–23). The two recorder series are **inner-joined on `timestamp`**, so only
hours present in *both* power and temperature survive. Rows with `mean is None`
are dropped before the join.

`add_lagged_features` then appends `power_lag_1..n` and `temp_lag_1..n` and drops
the leading `max(n_power_lags, n_temp_lags)` rows (which would hold NaN lags).

### Feature list (built in `coordinator.py`)

`get_default_features()` returns the base
`["year", "month", "day_of_week", "hour", "temperature"]`; the coordinator
appends `power_lag_1..n_power_lags` then `temp_lag_1..n_temp_lags`. The model is
trained and predicts on `df[features].values` in this exact column order, and
`predict_iterative` finds the power-lag columns by substring (`"power_lag" in
feat`) — so the names matter.

### Coordinator result dict (`_async_update_data` return → `coordinator.data`)

```python
{
  "predictions": [{"timestamp": iso_utc, "predicted": float}, ...],  # max_forecast_hours long
  "fitted":      [{"timestamp": iso_utc, "value": float}, ...],      # last 48 training hours
  "fitted_coverage": float,        # % of actuals <= fitted (state of the fitted sensor)
  "power_entity": str,
  "history_days": int,
  "max_forecast_hours": int,
  "last_updated": iso_local,       # dt_util.now()
  "training_samples": int,
}
```

`sensor.py` reads this dict. The 24h/48h sensors **slice** `predictions[:window]`;
the extended sensor uses the full list. Sensor state for the three forecast
sensors is `predictions[0]["predicted"]` (next-hour kW); the fitted sensor's
state is `fitted_coverage`. Display attributes convert the UTC `timestamp`s to
local time and rename keys to `time`/`value` (see `README.md` for the published
attribute format). If you change a key here, update the matching reader in
`sensor.py`.

## The model (`models.py`)

- **Algorithm:** quantile regression solved by **IRLS** (Iteratively Reweighted
  Least Squares) — asymmetric pinball-loss weights, solving
  `(XᵀWX + αI)β = XᵀWy` each iteration, warm-started from OLS. Pure numpy on
  purpose (lighter/faster than sklearn's LP solver for this data size). The
  intercept is an augmented leading ones column and is **not** regularised
  (`reg_diag[0] = 0`).
- **Dynamic peak/off-peak:** when `dynamic_config` is provided (it always is from
  the coordinator), `train` fits **two separate models** — one on hours where
  `peak_start <= hour <= peak_end`, one on the rest — and `predict` routes each
  row to the matching model by hour. The static single-quantile path
  (`self._coeffs`) still exists for `dynamic_config=None` but the coordinator
  does not use it.
- **Auto-regressive forecasting (`predict_iterative`):** future power lags are
  unknown, so it steps row-by-row, writing each prediction back into the
  `power_lag_*` columns of subsequent rows. If there are **no** power-lag
  features it short-circuits to a single vectorised `model.predict`. Temperature
  lags are *not* updated here — they're pre-seeded correctly in `_build_future_df`.
- `train`/`_train_dynamic` use `print(...)` for progress (visible in HA logs
  when training runs in the executor) — not the module logger. Leave as-is unless
  intentionally changing logging.

## `_build_future_df` (the trickiest function)

Builds the next-`n_hours` feature rows starting at the next full hour. Key points:

- **Timezone** is taken from the historical timestamps' `tzinfo` (recorder UTC).
- **Forecast lookup:** forecast entries are keyed by hour-truncated aware
  datetimes. **Timezone-naive forecast datetimes are assumed to be HA local
  time** (not UTC) — this handles integrations like BoM. Hours with no matching
  forecast fall back to `mean_temp_fallback` (the mean of historical
  temperature). Beyond weather availability, accuracy degrades — that's expected.
- **Lag seeding via concat+shift:** instead of manually seeding lag columns, it
  concatenates the historical tail with the future stub rows and calls
  `.shift(i)`. This makes the first future rows inherit correct lags from real
  history, and makes temperature lags track the evolving forecast hour-by-hour.
  Power lags seeded here are then overwritten by `predict_iterative`; temp lags
  are used as-is. Don't "simplify" this into per-row manual seeding without
  understanding why the concat approach is correct.

## Recorder statistics quirks

- `statistics_during_period(...)` is called with period `"hour"` and `{"mean"}`
  (native units). It returns a dict keyed by entity id.
- `_parse_start` handles **both** representations of a stat's `start`: a Unix
  epoch float (recent HA) and a datetime (older HA). Keep both branches.
- The coordinator raises `UpdateFailed` (not a bare exception) on any fetch/
  processing failure so HA shows a clean retry, and enforces
  `len(df) >= MIN_TRAINING_SAMPLES` (24).

## Config flow notes

- Step 1 (`user`) collects the optional integration name + the three required
  entities (power `sensor`, temperature `sensor`, weather `weather`). Step 2
  (`model`) collects model parameters. The options flow reuses the same
  `_model_schema` and only edits model parameters — **entity changes require
  re-adding the integration** (documented intent, not a bug).
- `NumberSelector` always returns floats; `_coerce_numbers` casts the integer
  fields back to `int` before saving. Add any new integer field to that set.
