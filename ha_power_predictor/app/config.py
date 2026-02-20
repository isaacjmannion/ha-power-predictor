"""
Configuration management for HA Power Predictor add-on.
Reads configuration from environment variables set by run.sh.
"""

import os
from typing import Dict, Any


def get_config_from_env() -> Dict[str, Any]:
    """
    Read configuration from environment variables.
    
    Returns:
        Dictionary with all configuration parameters
    """
    return {
        'power_entity': os.getenv('POWER_ENTITY', 'sensor.power_consumption'),
        'temperature_entity': os.getenv('TEMPERATURE_ENTITY', 'sensor.temperature'),
        'weather_forecast_entity': os.getenv('WEATHER_FORECAST_ENTITY', 'weather.home'),
        'bin_size_minutes': int(os.getenv('BIN_SIZE_MINUTES', '60')),
        'n_power_lags': int(os.getenv('N_POWER_LAGS', '10')),
        'n_temp_lags': int(os.getenv('N_TEMP_LAGS', '5')),
        'train_percentage': float(os.getenv('TRAIN_PERCENTAGE', '80.0')),
        'quantile': float(os.getenv('QUANTILE', '0.75')),
        'use_dynamic_quantile': os.getenv('USE_DYNAMIC_QUANTILE', 'true').lower() == 'true',
        'peak_start': int(os.getenv('PEAK_START', '9')),
        'peak_end': int(os.getenv('PEAK_END', '22')),
        'peak_quantile': float(os.getenv('PEAK_QUANTILE', '0.75')),
        'offpeak_quantile': float(os.getenv('OFFPEAK_QUANTILE', '0.50')),
        'history_days': int(os.getenv('HISTORY_DAYS', '90')),
        'timezone': os.getenv('TIMEZONE', 'UTC'),
        'use_hourly_statistics': os.getenv('USE_HOURLY_STATISTICS', 'true').lower() == 'true'
    }
