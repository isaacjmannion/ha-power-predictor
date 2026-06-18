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
4. Adds auto-regressive lag features (`add_lagged_features`).
5. Trains a `QuantileRegressionModel` (separate peak/off-peak models).
6. Computes in-sample fitted values + coverage % for charting.
7. Builds a future feature matrix out to `max_forecast_hours` (`_build_future_df`).
8. Generates predictions auto-regressively (`predict_iterative`).
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
  (selector + range), and both `strings.json` and `translations/en.json`.
  Caveat: `DEFAULT_UPDATE_INTERVAL_MINUTES` (10) is below the `config_flow`
  selector minimum (15) and is only ever hit as a code-level fallback — keep
  defaults within their selector ranges when editing. The README config table
  drifts from the code in several places (update interval, min/max power
  defaults) **and** still documents removed options (`Use Dynamic Quantile`,
  `Quantile`) that `config_flow.py` no longer exposes — trust
  `const.py`/`config_flow.py`, and fix the README if you touch defaults.
- **`strings.json` and `translations/en.json` must stay in sync.** `strings.json`
  is the source; `en.json` is its English copy. They currently differ slightly
  (some legacy `use_dynamic_quantile`/`quantile` strings remain that the config
  flow no longer exposes — peak/off-peak quantiles are always active).
- **Times are UTC inside the pipeline; local only at the edge.** Statistics,
  features, and prediction timestamps are UTC-aware throughout. Conversion to
  the user's local timezone happens only in `sensor.py` for display attributes.
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
  pure modules import no Home Assistant). HA-harness tests are not set up yet.
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

> `zip_release`/`filename` in `hacs.json` make HACS install from that zip asset;
> only enable them once a release actually carries the zip.

## Things that must stay consistent

- Config keys/defaults in `const.py` ⇄ selectors in `config_flow.py` ⇄ labels in
  `strings.json` ⇄ `translations/en.json`.
- Sensor/attribute shapes documented in `README.md` ⇄ what `sensor.py` actually
  emits.
- `DOMAIN` ("ha_power_predictor") ⇄ the folder name ⇄ `domain` in
  `manifest.json`.
