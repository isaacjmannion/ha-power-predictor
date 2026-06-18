"""Home Assistant harness tests for the config and options flows.

These run under pytest-homeassistant-custom-component (the `hass` and
`enable_custom_integrations` fixtures), so they exercise the real flow code —
including the hour-offsets ObjectSelector, which the pure tests can't reach.
"""

from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ha_power_predictor.config_flow import _hour_offsets_error
from custom_components.ha_power_predictor.const import DOMAIN
from custom_components.ha_power_predictor.data_processing import normalize_hour_offsets

# Every vol.Required field in _model_schema must be supplied when submitting.
MODEL_PARAMS = {
    "min_power": 0.5,
    "max_power": 15.0,
    "history_days": 30,
    "update_interval_minutes": 60,
    "n_power_lags": 5,
    "n_temp_lags": 5,
    "peak_start": 9,
    "peak_end": 22,
    "peak_quantile": 0.75,
    "offpeak_quantile": 0.5,
    "max_forecast_hours": 48,
}

ENTRY_DATA = {
    "integration_name": "Test Predictor",
    "power_entity": "sensor.power",
    "temperature_entity": "sensor.temperature",
    "weather_forecast_entity": "weather.home",
    **MODEL_PARAMS,
}


async def test_user_step_shows_form(hass):
    """The first config step renders the entity-selection form."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"


async def test_options_flow_form_renders(hass):
    """The options form builds — this constructs the hour-offsets ObjectSelector."""
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA)
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "init"


async def test_options_flow_saves_hour_offsets(hass):
    """Submitting hour offsets through the options flow stores them on the entry."""
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA)
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    user_input = {**MODEL_PARAMS, "hour_offsets": [{"hour": 18, "offset": 0.8}]}
    result = await hass.config_entries.options.async_configure(result["flow_id"], user_input)
    assert result["type"] == FlowResultType.CREATE_ENTRY
    # NumberSelector may return floats (hour 18.0); normalize to compare robustly.
    assert normalize_hour_offsets(result["data"]["hour_offsets"]) == {18: 0.8}


def test_hour_offsets_error_accepts_valid_and_rejects_invalid():
    """The flow-level validator passes good rows and flags bad ones."""
    assert _hour_offsets_error({}) is None
    assert _hour_offsets_error({"hour_offsets": []}) is None
    assert _hour_offsets_error({"hour_offsets": [{"hour": 18, "offset": 0.8}]}) is None

    bad_hour = {"hour_offsets": [{"hour": 99, "offset": 1.0}]}
    assert _hour_offsets_error(bad_hour) == "invalid_hour_offsets"

    bad_offset = {"hour_offsets": [{"hour": 5, "offset": "x"}]}
    assert _hour_offsets_error(bad_offset) == "invalid_hour_offsets"
