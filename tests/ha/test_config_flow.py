"""Home Assistant harness tests.

These run under pytest-homeassistant-custom-component on the pinned HA version,
so they import the integration under *real* Home Assistant. That catches
import-time breakage and — most importantly — validates that the config schema,
including the hour-offsets ObjectSelector (which needs HA 2025.7+), is accepted
by this HA version. They are plain sync tests: no running hass or recorder setup
is required, which keeps them fast and robust.
"""

from custom_components.ha_power_predictor.config_flow import _hour_offsets_error, _model_schema
from custom_components.ha_power_predictor.const import CONF_HOUR_OFFSETS


def test_model_schema_builds_and_includes_hour_offsets():
    # Calling _model_schema constructs every selector, including the hour-offsets
    # ObjectSelector — if its config were invalid for this HA version, this raises.
    schema = _model_schema({})
    keys = {marker.schema for marker in schema.schema}
    assert CONF_HOUR_OFFSETS in keys


def test_hour_offsets_error_accepts_valid_and_rejects_invalid():
    assert _hour_offsets_error({}) is None
    assert _hour_offsets_error({"hour_offsets": []}) is None
    assert _hour_offsets_error({"hour_offsets": [{"hour": 18, "offset": 0.8}]}) is None

    bad_hour = {"hour_offsets": [{"hour": 99, "offset": 1.0}]}
    assert _hour_offsets_error(bad_hour) == "invalid_hour_offsets"

    bad_offset = {"hour_offsets": [{"hour": 5, "offset": "x"}]}
    assert _hour_offsets_error(bad_offset) == "invalid_hour_offsets"
