# CLAUDE.md

Guidance for Claude Code (and other AI agents) working in this repository.

## What this is

**HA Power Predictor** is a custom [Home Assistant](https://www.home-assistant.io/)
integration, distributed via [HACS](https://hacs.xyz/). It forecasts future
household power consumption from historical usage + temperature (pulled from the
HA recorder's long-term statistics) plus a weather forecast, using a
quantile-regression model. Predictions are published as HA sensor entities.

There is **no standalone application**: the code only runs inside a Home
Assistant instance. All the Python lives in one integration package under
`custom_components/ha_power_predictor/`.

## Repository layout

```
custom_components/ha_power_predictor/   # the integration (all code lives here)
  __init__.py          # setup/teardown of the config entry
  coordinator.py       # the pipeline: fetch → process → train → predict
  models.py            # pure-numpy IRLS quantile regression + iterative forecast
  data_processing.py   # recorder statistics → feature DataFrame + lag features
  config_flow.py       # 2-step UI setup + options flow
  sensor.py            # 4 sensor entities (24h / 48h / extended / fitted)
  button.py            # "Train Now" button
  const.py             # DOMAIN, config keys, and ALL default values
  manifest.json        # HA integration manifest (version, requirements, deps)
  strings.json         # UI strings (source of truth for translations)
  translations/en.json # English translation of strings.json
  brand/               # icon/logo PNGs shown in the HA UI
hacs.json              # HACS metadata
repository.yaml        # HACS repository descriptor
README.md              # user-facing docs (install, config, dashboard example)
images/                # README screenshots and banners
pyproject.toml         # ruff (lint) + pytest config
tests/pure/            # pytest suite for the HA-independent modules
tests/ha/              # HA-harness tests (pytest-homeassistant-custom-component)
requirements-test.txt  # pinned HA + PHACC for the harness job
.github/workflows/     # CI/CD: validate.yml, lint.yml, test.yml, release.yml
```

See `custom_components/ha_power_predictor/CLAUDE.md` for module-level internals
(data contracts, the IRLS algorithm, auto-regressive inference, timezone rules).

## Architecture / data flow

A single `DataUpdateCoordinator` (`PowerPredictorCoordinator` in
`coordinator.py`) drives the whole pipeline on a timer (the configured update
interval) and on demand (the "Train Now" button). On each run
(`_async_update_data`) it:

1. Fetches hourly recorder statistics (`mean`) for the power and temperature
   entities over the last `history_days`.
2. Fetches the hourly weather forecast via the `weather.get_forecasts` service.
3. Processes the statistics into a feature DataFrame (`process_ha_statistics`).
4. Adds auto-regressive lag features (`add_lagged_features`) and cyclical
   hour-of-day features (`add_cyclical_features`).
5. Trains a `QuantileRegressionModel` (separate peak/off-peak models) on
   standardized features, with a per-feature ridge penalty from the configured
   influence weights (`build_feature_weights`). Peak/off-peak routing uses the
   **local** hour-of-day. Also trains a second q=0.5 `state_model` (the median),
   used to seed the lag columns during forecasting.
6. Computes in-sample fitted values + coverage % for charting.
7. Builds a future feature matrix out to `max_forecast_hours` (`_build_future_df`).
8. Generates predictions auto-regressively (`predict_iterative`), feeding the
   `state_model`'s median back into the power lags (so the conservative quantile
   does not compound over the horizon) while reporting the quantile model, then
   adds any configured per-hour offsets (still clamped to min/max power).
9. Returns a dict that the sensor entities read.

The sensors (`sensor.py`) are thin `CoordinatorEntity` wrappers — they hold no
logic of their own beyond slicing/formatting the coordinator's result dict. The
button (`button.py`) is a plain `ButtonEntity` (not a `CoordinatorEntity`) that
just calls `coordinator.async_request_refresh()` on press.

## Conventions & gotchas (read before editing Python)

- **Never block the event loop.** All CPU-bound work (pandas processing, model
  training, prediction, building the future matrix) is dispatched to the
  executor via `hass.async_add_executor_job(...)` (recorder reads use
  `get_instance(hass).async_add_executor_job`). Keep it that way — adding heavy
  synchronous work directly in an `async def` is a bug.
- **Config precedence is `{**entry.data, **entry.options}`** everywhere config
  is read (options override the values captured at setup). Reuse this pattern;
  don't read `entry.data` alone for tunable parameters.
- **`const.py` is the single source of truth for defaults.** When you add or
  change a config field, update `const.py` (key + default), `config_flow.py`
  (selector + range), and both `strings.json` and `translations/en.json`; keep
  defaults within their selector ranges. The README's "Model Parameters" table
  drifts from the code in places (its update-interval / min-power / max-power
  defaults are stale, and it still lists removed options `Use Dynamic Quantile` /
  `Quantile` that `config_flow.py` no longer exposes) — trust
  `const.py`/`config_flow.py`, and fix the README if you touch defaults.
- **`strings.json` and `translations/en.json` must stay in sync.** `strings.json`
  is the source; `en.json` is its English copy. They currently differ slightly
  (some legacy `use_dynamic_quantile`/`quantile` strings remain that the config
  flow no longer exposes — peak/off-peak quantiles are always active).
- **Times are UTC inside the pipeline; local only at the edge.** Statistics,
  features, and prediction timestamps are UTC-aware throughout, and the model's
  cyclical `hour` feature keys off **UTC** hours. Three places use **local**
  time: `sensor.py` (display attributes), the **hour-of-day offset** lookup, and
  the **peak/off-peak routing** (`dt_util.as_local(ts).hour` / `tz_convert`, so
  an offset or a peak window for hour 13 lands at 1 pm on the user's clock). See
  the package CLAUDE.md "Hour-of-day offsets".
- **No new heavy dependencies.** The model is deliberately pure-numpy (no
  scikit-learn / scipy) so the integration stays light. Declared requirements
  are `numpy>=1.24.0` and `pandas>=2.0.0`, set in `manifest.json` — the single
  source of truth for `version`, `requirements`, and `dependencies`. **Do not put
  those keys in `hacs.json`:** the HACS validation action rejects any key outside
  its own schema (`name`, `homeassistant`, `zip_release`, `filename`, …), so
  `hacs.json` holds HACS metadata only.

## Validating changes

CI runs automatically on every push/PR to `main` (see `.github/workflows/`):
**hassfest** + **HACS** validation (`validate.yml`), **ruff** lint (`lint.yml`),
and the **pytest** suite (`test.yml`). Reproduce them locally:

- **Lint:** `ruff check .` (config in `pyproject.toml`; lenient set `E/F/W/I`).
- **Tests:** `pytest tests/pure` (needs only `pytest`, `numpy`, `pandas` — the
  pure modules import no Home Assistant). `pytest tests/ha` runs the HA-harness
  tests under `pytest-homeassistant-custom-component` (pinned in
  `requirements-test.txt`, Python 3.14); they import the integration under real
  Home Assistant. CI runs both as separate jobs in `test.yml`.
- **JSON validity** for `manifest.json`, `hacs.json`, `strings.json`,
  `translations/en.json` (e.g. `python -m json.tool <file>`).
- **Manual run:** copy `custom_components/ha_power_predictor/` into a Home
  Assistant `config/custom_components/` directory, restart HA, add the
  integration via the UI, and watch the logs (the coordinator logs each pipeline
  stage at `debug`/`info`).

`hassfest` and the HACS action are the canonical structural checks and now run
in CI. There is still no automated way to verify model *output* without a live
HA instance with recorder statistics — reason about model changes from the code
and the docstrings in `models.py`.

## Releasing / bumping the version

`version` in `manifest.json` is the single source of truth (both HA and HACS
read it). `hacs.json` must NOT carry a `version` key — the HACS validation
action rejects it. When releasing:

1. Bump `version` in `manifest.json`.
2. Add a `### X.Y.Z — title` entry to the Changelog in `README.md`.
3. PR → confirm CI is green → merge to `main`.
4. Publish a GitHub release with tag `vX.Y.Z` (use the `v` prefix consistently).
   `release.yml` then stamps the tag's version into the manifest and attaches
   `ha_power_predictor.zip` to the release. HACS serves the latest release.

> `zip_release: true` + `filename: "ha_power_predictor.zip"` in `hacs.json` are
> **active** — HACS installs from the zip asset `release.yml` attaches (faster,
> atomic). Keep `filename` matching the zip name the workflow builds.

## Things that must stay consistent

- Config keys/defaults in `const.py` ⇄ selectors in `config_flow.py` ⇄ labels in
  `strings.json` ⇄ `translations/en.json`.
- Sensor/attribute shapes documented in `README.md` ⇄ what `sensor.py` actually
  emits.
- `DOMAIN` ("ha_power_predictor") ⇄ the folder name ⇄ `domain` in
  `manifest.json`.
- Minimum HA version: `homeassistant` in `hacs.json` ⇄ README "Requirements"
  (currently **2025.7** — the hour-offset editor uses the object-selector
  `fields`/`multiple` row form added in HA 2025.7; don't drop below it without
  changing that selector).
- `filename` in `hacs.json` ⇄ the zip name `release.yml` builds
  (`ha_power_predictor.zip`).
