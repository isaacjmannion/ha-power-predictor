"""
Config flow for HA Power Predictor.

Step 1 (user):  Select the three required entities using HA's entity picker.
Step 2 (model): Tune model parameters (all have sensible defaults).

An options flow is also provided so parameters can be changed after setup
without removing and re-adding the integration. Saving options triggers an
automatic entry reload so the new settings take effect immediately.
"""

from __future__ import annotations

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_HISTORY_DAYS,
    CONF_INTEGRATION_NAME,
    CONF_MAX_FORECAST_HOURS,
    CONF_MAX_POWER,
    CONF_MIN_POWER,
    CONF_N_POWER_LAGS,
    CONF_N_TEMP_LAGS,
    CONF_OFFPEAK_QUANTILE,
    CONF_PEAK_END,
    CONF_PEAK_QUANTILE,
    CONF_PEAK_START,
    CONF_POWER_ENTITY,
    CONF_TEMPERATURE_ENTITY,
    CONF_UPDATE_INTERVAL_MINUTES,
    CONF_WEATHER_FORECAST_ENTITY,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_INTEGRATION_NAME,
    DEFAULT_MAX_FORECAST_HOURS,
    DEFAULT_MAX_POWER,
    DEFAULT_MIN_POWER,
    DEFAULT_N_POWER_LAGS,
    DEFAULT_N_TEMP_LAGS,
    DEFAULT_OFFPEAK_QUANTILE,
    DEFAULT_PEAK_END,
    DEFAULT_PEAK_QUANTILE,
    DEFAULT_PEAK_START,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    DOMAIN,
    MAX_FORECAST_HOURS_LIMIT,
)

# ── Step 1 schema: entity selectors ─────────────────────────────────────────

STEP_ENTITIES_SCHEMA = vol.Schema({
    vol.Optional(CONF_INTEGRATION_NAME, default=DEFAULT_INTEGRATION_NAME): selector.TextSelector(),
    vol.Required(CONF_POWER_ENTITY): selector.EntitySelector(
        selector.EntitySelectorConfig(domain="sensor")
    ),
    vol.Required(CONF_TEMPERATURE_ENTITY): selector.EntitySelector(
        selector.EntitySelectorConfig(domain="sensor")
    ),
    vol.Required(CONF_WEATHER_FORECAST_ENTITY): selector.EntitySelector(
        selector.EntitySelectorConfig(domain="weather")
    ),
})


# ── Step 2 / options schema: model parameters ────────────────────────────────

def _model_schema(defaults: dict) -> vol.Schema:
    """
    Build the model parameter schema, pre-populated with current values.

    NumberSelector returns floats, so integer fields are coerced via
    _coerce_numbers() before saving.
    """
    def _d(key, fallback):
        return defaults.get(key, fallback)

    return vol.Schema({
        vol.Required(
            CONF_MIN_POWER,
            default=_d(CONF_MIN_POWER, DEFAULT_MIN_POWER),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0.0, max=100.0, step=0.1, mode="box")
        ),
        vol.Required(
            CONF_MAX_POWER,
            default=_d(CONF_MAX_POWER, DEFAULT_MAX_POWER),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0.0, max=1000.0, step=0.1, mode="box")
        ),
        vol.Required(
            CONF_HISTORY_DAYS,
            default=_d(CONF_HISTORY_DAYS, DEFAULT_HISTORY_DAYS),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=1, max=365, step=1, mode="box")
        ),
        vol.Required(
            CONF_UPDATE_INTERVAL_MINUTES,
            default=_d(CONF_UPDATE_INTERVAL_MINUTES, DEFAULT_UPDATE_INTERVAL_MINUTES),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=5, max=1440, step=5, mode="box")
        ),
        vol.Required(
            CONF_N_POWER_LAGS,
            default=_d(CONF_N_POWER_LAGS, DEFAULT_N_POWER_LAGS),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=50, step=1, mode="box")
        ),
        vol.Required(
            CONF_N_TEMP_LAGS,
            default=_d(CONF_N_TEMP_LAGS, DEFAULT_N_TEMP_LAGS),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=20, step=1, mode="box")
        ),
        vol.Required(
            CONF_PEAK_START,
            default=_d(CONF_PEAK_START, DEFAULT_PEAK_START),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=23, step=1, mode="box")
        ),
        vol.Required(
            CONF_PEAK_END,
            default=_d(CONF_PEAK_END, DEFAULT_PEAK_END),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=23, step=1, mode="box")
        ),
        vol.Required(
            CONF_PEAK_QUANTILE,
            default=_d(CONF_PEAK_QUANTILE, DEFAULT_PEAK_QUANTILE),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0.5, max=0.99, step=0.01, mode="slider")
        ),
        vol.Required(
            CONF_OFFPEAK_QUANTILE,
            default=_d(CONF_OFFPEAK_QUANTILE, DEFAULT_OFFPEAK_QUANTILE),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0.5, max=0.99, step=0.01, mode="slider")
        ),
        vol.Required(
            CONF_MAX_FORECAST_HOURS,
            default=_d(CONF_MAX_FORECAST_HOURS, DEFAULT_MAX_FORECAST_HOURS),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=48, max=MAX_FORECAST_HOURS_LIMIT, step=24, mode="box")
        ),
    })


# ── Config flow ──────────────────────────────────────────────────────────────

class PowerPredictorConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle initial setup via the HA UI."""

    VERSION = 1

    def __init__(self) -> None:
        self._entity_data: dict = {}

    async def async_step_user(
        self,
        user_input: dict | None = None,
    ) -> config_entries.FlowResult:
        """Step 1 — entity selection."""
        errors: dict = {}

        if user_input is not None:
            self._entity_data = user_input
            return await self.async_step_model()

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_ENTITIES_SCHEMA,
            errors=errors,
        )

    async def async_step_model(
        self,
        user_input: dict | None = None,
    ) -> config_entries.FlowResult:
        """Step 2 — model parameters."""
        errors: dict = {}

        if user_input is not None:
            data = {**self._entity_data, **_coerce_numbers(user_input)}
            title = self._entity_data.get(CONF_INTEGRATION_NAME, DEFAULT_INTEGRATION_NAME)
            return self.async_create_entry(title=title, data=data)

        return self.async_show_form(
            step_id="model",
            data_schema=_model_schema({}),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> "PowerPredictorOptionsFlow":
        """Return the options flow handler."""
        return PowerPredictorOptionsFlow(config_entry)


# ── Options flow ─────────────────────────────────────────────────────────────

class PowerPredictorOptionsFlow(config_entries.OptionsFlow):
    """
    Allow the user to adjust model parameters after setup.

    Saving triggers an entry reload (via the listener in __init__.py) so
    changes — including a new update interval — take effect immediately.
    Entity selections are intentionally excluded from the options flow;
    to change entities the user should delete and re-add the integration.
    """

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(
        self,
        user_input: dict | None = None,
    ) -> config_entries.FlowResult:
        """Single options step — all model parameters on one screen."""
        if user_input is not None:
            return self.async_create_entry(title="", data=_coerce_numbers(user_input))

        # Pre-populate with existing values (options override data)
        current = {**self._config_entry.data, **self._config_entry.options}

        return self.async_show_form(
            step_id="init",
            data_schema=_model_schema(current),
        )


# ── Helper ───────────────────────────────────────────────────────────────────

def _coerce_numbers(data: dict) -> dict:
    """
    NumberSelector always returns floats. Coerce fields that should be integers.
    """
    int_fields = {
        CONF_HISTORY_DAYS,
        CONF_UPDATE_INTERVAL_MINUTES,
        CONF_N_POWER_LAGS,
        CONF_N_TEMP_LAGS,
        CONF_PEAK_START,
        CONF_PEAK_END,
        CONF_MAX_FORECAST_HOURS,
    }
    return {k: int(v) if k in int_fields else v for k, v in data.items()}
