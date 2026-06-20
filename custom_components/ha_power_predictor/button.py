"""
Button platform for HA Power Predictor.

Registers per-instance action buttons:
  - "Train Now"            — triggers an immediate full pipeline run.
  - "Export Training Data" — exports the configured history window + config.
  - "Export All History"   — exports up to a year of recorder data + config.

The export buttons write a JSON file to the HA config directory for offline
analysis / backtesting (see the coordinator's async_export_data).
"""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_HISTORY_DAYS,
    CONF_INTEGRATION_NAME,
    DEFAULT_EXPORT_FULL_DAYS,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_INTEGRATION_NAME,
    DOMAIN,
)
from .coordinator import PowerPredictorCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Register the Train Now + data-export buttons for this config entry."""
    coordinator: PowerPredictorCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([
        TrainNowButton(coordinator, entry),
        ExportDataButton(coordinator, entry, "training"),
        ExportDataButton(coordinator, entry, "full"),
    ])


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


class ExportDataButton(ButtonEntity):
    """
    Button that exports raw training data + the resolved config to a JSON file.

    Two instances are registered per config entry:
      - scope "training" exports the configured History Days window.
      - scope "full" exports up to DEFAULT_EXPORT_FULL_DAYS of recorder data.

    The file is written to the HA config directory by the coordinator; a
    persistent notification reports the path.
    """

    _attr_icon = "mdi:database-export"
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: PowerPredictorCoordinator,
        entry: ConfigEntry,
        scope: str,
    ) -> None:
        self._coordinator = coordinator
        self._entry = entry
        self._scope = scope  # "training" or "full"
        self._attr_unique_id = f"{entry.entry_id}_export_{scope}"

        integration_name = entry.data.get(CONF_INTEGRATION_NAME, DEFAULT_INTEGRATION_NAME)
        label = "Export Training Data" if scope == "training" else "Export All History"
        self._attr_name = f"{integration_name} {label}"

    async def async_press(self) -> None:
        """Export recorder data + config to a JSON file in the config dir."""
        if self._scope == "training":
            cfg = {**self._entry.data, **self._entry.options}
            days = int(cfg.get(CONF_HISTORY_DAYS, DEFAULT_HISTORY_DAYS))
        else:
            days = DEFAULT_EXPORT_FULL_DAYS
        await self._coordinator.async_export_data(days, self._scope)
