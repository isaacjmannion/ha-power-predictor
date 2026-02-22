"""Constants for HA Power Predictor."""

DOMAIN = "ha_power_predictor"

PLATFORMS = ["sensor", "button"]

# Config entry keys
CONF_POWER_ENTITY = "power_entity"
CONF_TEMPERATURE_ENTITY = "temperature_entity"
CONF_WEATHER_FORECAST_ENTITY = "weather_forecast_entity"
CONF_N_POWER_LAGS = "n_power_lags"
CONF_N_TEMP_LAGS = "n_temp_lags"
CONF_QUANTILE = "quantile"
CONF_USE_DYNAMIC_QUANTILE = "use_dynamic_quantile"
CONF_PEAK_START = "peak_start"
CONF_PEAK_END = "peak_end"
CONF_PEAK_QUANTILE = "peak_quantile"
CONF_OFFPEAK_QUANTILE = "offpeak_quantile"
CONF_HISTORY_DAYS = "history_days"
CONF_UPDATE_INTERVAL_MINUTES = "update_interval_minutes"

# Defaults
DEFAULT_N_POWER_LAGS = 5
DEFAULT_N_TEMP_LAGS = 5
DEFAULT_QUANTILE = 0.75
DEFAULT_USE_DYNAMIC_QUANTILE = True
DEFAULT_PEAK_START = 9
DEFAULT_PEAK_END = 22
DEFAULT_PEAK_QUANTILE = 0.75
DEFAULT_OFFPEAK_QUANTILE = 0.50
DEFAULT_HISTORY_DAYS = 30
DEFAULT_UPDATE_INTERVAL_MINUTES = 60

# Pipeline constants
MIN_TRAINING_SAMPLES = 24  # 1 day of hourly statistics
PREDICTION_HOURS = 48
