"""Constants for HA Power Predictor."""

DOMAIN = "ha_power_predictor_dev"

PLATFORMS = ["sensor", "button"]

# Config entry keys
CONF_INTEGRATION_NAME = "integration_name"
CONF_POWER_ENTITY = "power_entity"
CONF_TEMPERATURE_ENTITY = "temperature_entity"
CONF_WEATHER_FORECAST_ENTITY = "weather_forecast_entity"
CONF_N_POWER_LAGS = "n_power_lags"
CONF_N_TEMP_LAGS = "n_temp_lags"
CONF_PEAK_START = "peak_start"
CONF_PEAK_END = "peak_end"
CONF_PEAK_QUANTILE = "peak_quantile"
CONF_OFFPEAK_QUANTILE = "offpeak_quantile"
CONF_HISTORY_DAYS = "history_days"
CONF_UPDATE_INTERVAL_MINUTES = "update_interval_minutes"
CONF_MIN_POWER = "min_power"
CONF_MAX_POWER = "max_power"
CONF_MAX_FORECAST_HOURS = "max_forecast_hours"
CONF_HOUR_OFFSETS = "hour_offsets"
CONF_HOUR_HARMONICS = "hour_harmonics"
CONF_REG_ALPHA = "reg_alpha"
CONF_WEIGHT_TIME = "weight_time"
CONF_WEIGHT_TEMPERATURE = "weight_temperature"
CONF_WEIGHT_LAGS = "weight_lags"

# Defaults
DEFAULT_INTEGRATION_NAME = "Power Predictor Dev"
DEFAULT_N_POWER_LAGS = 5
DEFAULT_N_TEMP_LAGS = 0  # 0 by default (temperature is off by default; see weights below)
DEFAULT_PEAK_START = 9
DEFAULT_PEAK_END = 22
DEFAULT_PEAK_QUANTILE = 0.75
DEFAULT_OFFPEAK_QUANTILE = 0.50
DEFAULT_HISTORY_DAYS = 30
DEFAULT_UPDATE_INTERVAL_MINUTES = 10
DEFAULT_MIN_POWER = 0.5
DEFAULT_MAX_POWER = 15.0
DEFAULT_MAX_FORECAST_HOURS = 48  # 2 days
DEFAULT_HOUR_OFFSETS: list = []  # rows of {"hour": int, "offset": float kW}
# Defaults below were retuned from walk-forward backtesting (see tools/cv_sweep.py).
DEFAULT_HOUR_HARMONICS = 3  # sin/cos harmonics for hour-of-day (0 = linear hour)
DEFAULT_REG_ALPHA = 0.1  # L2 strength on standardized features (higher over-smooths)
DEFAULT_WEIGHT_TIME = 2.0  # influence of time-of-day features (1.0 = neutral)
DEFAULT_WEIGHT_TEMPERATURE = 0.0  # temperature off by default; raise if it helps your load
DEFAULT_WEIGHT_LAGS = 0.5  # influence of power-lag features

# Pipeline constants
MIN_TRAINING_SAMPLES = 24  # 1 day of hourly statistics
MAX_FORECAST_HOURS_LIMIT = 168  # 7 days maximum

# Data export (for offline analysis / backtesting)
EXPORT_SCHEMA_VERSION = 1
DEFAULT_EXPORT_FULL_DAYS = 365  # "all available" export window (recorder cap)
