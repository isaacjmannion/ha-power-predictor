"""Fixtures for the Home Assistant harness tests (tests/ha)."""

import pytest


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Let HA discover and load the ha_power_predictor custom integration in tests.

    Without this, Home Assistant refuses to load anything under custom_components/,
    so the config/options flows can't be found. The fixture is provided by
    pytest-homeassistant-custom-component.
    """
    yield
