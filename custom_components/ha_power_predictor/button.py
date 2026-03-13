"""
Button platform for HA Power Predictor.

Registers a single "Train Now" button that immediately triggers a full
pipeline run via the coordinator, regardless of the scheduled update interval.
This is useful after changing configuration or after a long gap in data.
"""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_INTEGRATION_NAME, DEFAULT_INTEGRATION_NAME, DOMAIN
from .coordinator import PowerPredictorCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Register the Train Now button for this config entry."""
    coordinator: PowerPredictorCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([TrainNowButton(coordinator, entry)])


class TrainNowButton(ButtonEntity):
    """
    Button that triggers an immediate model retrain.

    Pressing this button calls coordinator.async_request_refresh(), which
    runs the full fetch → train → predict pipeline and updates both sensors.
    The scheduled update interval is unaffected.
    """

    _attr_icon = "mdi:brain"
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: PowerPredictorCoordinator,
        entry: ConfigEntry,
    ) -> None:
        self._coordinator = coordinator
        self._attr_unique_id = f"{entry.entry_id}_train_now"

        # Get integration name from config, falling back to default
        integration_name = entry.data.get(CONF_INTEGRATION_NAME, DEFAULT_INTEGRATION_NAME)
        self._attr_name = f"{integration_name} Train Now"

    async def async_press(self) -> None:
        """Immediately trigger a full pipeline run."""
        await self._coordinator.async_request_refresh()
