"""
Sensor platform for HA Power Predictor.

Registers two sensors:
  - sensor.power_prediction_24h  — covers the next 24 hours
  - sensor.power_prediction_48h  — covers the next 48 hours

Both sensors share the same coordinator data. Their state is the predicted
power value for the next (current) hour; the full hourly breakdown is exposed
as an attribute so it can be used in Lovelace charts and automations.
"""

from __future__ import annotations

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import PowerPredictorCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the two prediction sensors from a config entry."""
    coordinator: PowerPredictorCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([
        PowerPredictionSensor(coordinator, entry, window_hours=24),
        PowerPredictionSensor(coordinator, entry, window_hours=48),
    ])


class PowerPredictionSensor(CoordinatorEntity[PowerPredictorCoordinator], SensorEntity):
    """
    A sensor exposing the quantile-regression power prediction for a given
    forecast window (24 h or 48 h).

    State
    -----
    The predicted kW value for the *next* full hour — i.e. predictions[0].
    Both the 24h and 48h sensors share this same immediate next-hour value
    as their state; they differ only in how many hours their attribute list spans.

    Attributes
    ----------
    hourly_predictions  List of {"timestamp": ISO-8601, "predicted": kW} dicts
                        covering the full window (24 or 48 entries).
    peak                Highest predicted value in the window (kW).
    average             Mean predicted value across the window (kW).
    window_hours        Integer window length for this sensor.
    forecast_start      ISO-8601 timestamp of the first prediction.
    forecast_end        ISO-8601 timestamp of the last prediction.
    source_entity       The power consumption entity used to train the model.
    last_trained        ISO-8601 timestamp of the last completed pipeline run.
    """

    _attr_device_class = SensorDeviceClass.POWER
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "kW"
    _attr_icon = "mdi:lightning-bolt"
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: PowerPredictorCoordinator,
        entry: ConfigEntry,
        window_hours: int,
    ) -> None:
        super().__init__(coordinator)
        self._window_hours = window_hours
        self._attr_unique_id = f"{entry.entry_id}_prediction_{window_hours}h"
        self._attr_name = f"Power Prediction {window_hours}h"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _window_predictions(self) -> list[dict]:
        """Slice of the coordinator predictions for this window."""
        if not self.coordinator.data:
            return []
        return self.coordinator.data.get("predictions", [])[: self._window_hours]

    # ------------------------------------------------------------------
    # SensorEntity interface
    # ------------------------------------------------------------------

    @property
    def native_value(self) -> float | None:
        """Current state — predicted power for the next hour."""
        preds = self._window_predictions
        if not preds:
            return None
        return preds[0]["predicted"]

    @property
    def extra_state_attributes(self) -> dict:
        """Rich attributes for charting and automations."""
        preds = self._window_predictions
        if not preds:
            return {}

        predicted_values = [p["predicted"] for p in preds]

        return {
            "hourly_predictions": preds,
            "window_hours": self._window_hours,
            "peak": round(max(predicted_values), 3),
            "average": round(sum(predicted_values) / len(predicted_values), 3),
            "forecast_start": preds[0]["timestamp"],
            "forecast_end": preds[-1]["timestamp"],
            "source_entity": self.coordinator.data.get("power_entity"),
            "last_trained": self.coordinator.data.get("last_updated"),
        }
