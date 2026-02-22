"""
Sensor platform for HA Power Predictor.

Registers two sensors:
  - sensor.power_prediction_24h  — covers the next 24 hours
  - sensor.power_prediction_48h  — covers the next 48 hours

Attribute format:
  source_entity        str  — entity ID used for training
  history_days         int  — days of history the model was trained on
  last_forecast_update str  — human-readable local datetime of last pipeline run
  forecast             list — [{time: ISO-8601 local, value: kW}, ...]
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
from homeassistant.util import dt as dt_util

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
    Power prediction sensor for a given forecast window (24 h or 48 h).

    State
    -----
    Predicted kW for the next full hour (predictions[0]).

    Attributes
    ----------
    source_entity        Entity ID of the power consumption sensor used for training.
    history_days         Number of days of history the model trained on.
    last_forecast_update Friendly local-time string of when the pipeline last ran.
    forecast             List of {time, value} dicts for each hour in the window.
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

    @property
    def _window_predictions(self) -> list[dict]:
        """Coordinator predictions sliced to this sensor's window."""
        if not self.coordinator.data:
            return []
        return self.coordinator.data.get("predictions", [])[: self._window_hours]

    @property
    def native_value(self) -> float | None:
        """State — predicted kW for the next hour."""
        preds = self._window_predictions
        return preds[0]["predicted"] if preds else None

    @property
    def extra_state_attributes(self) -> dict:
        """Attributes matching the documented format."""
        data = self.coordinator.data
        if not data:
            return {}

        preds = self._window_predictions
        if not preds:
            return {}

        # Format last_updated as a friendly local datetime string
        last_updated_raw = data.get("last_updated", "")
        try:
            last_updated_local = dt_util.as_local(
                dt_util.parse_datetime(last_updated_raw)
            ).strftime("%B %-d, %Y at %H:%M:%S")
        except (TypeError, ValueError, AttributeError):
            last_updated_local = last_updated_raw

        # Build forecast list: {time: ISO local with offset, value: kW}
        forecast = []
        for p in preds:
            try:
                utc_dt = dt_util.parse_datetime(p["timestamp"])
                local_dt = dt_util.as_local(utc_dt)
                forecast.append({
                    "time": local_dt.isoformat(),
                    "value": p["predicted"],
                })
            except (TypeError, ValueError, AttributeError):
                continue

        return {
            "source_entity": data.get("power_entity"),
            "history_days": data.get("history_days"),
            "last_forecast_update": last_updated_local,
            "forecast": forecast,
        }
