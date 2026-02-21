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
        'bin_size_minutes': int(os.getenv('BIN_SIZE_MINUTES') or '60'),
        'n_power_lags': int(os.getenv('N_POWER_LAGS') or '10'),
        'n_temp_lags': int(os.getenv('N_TEMP_LAGS') or '5'),
        'train_percentage': float(os.getenv('TRAIN_PERCENTAGE') or '80.0'),
        'quantile': float(os.getenv('QUANTILE') or '0.75'),
        'use_dynamic_quantile': (os.getenv('USE_DYNAMIC_QUANTILE') or 'true').lower() == 'true',
        'peak_start': int(os.getenv('PEAK_START') or '9'),
        'peak_end': int(os.getenv('PEAK_END') or '22'),
        'peak_quantile': float(os.getenv('PEAK_QUANTILE') or '0.75'),
        'offpeak_quantile': float(os.getenv('OFFPEAK_QUANTILE') or '0.50'),
        'history_days': int(os.getenv('HISTORY_DAYS') or '30'),
        'timezone': os.getenv('TIMEZONE') or 'UTC',
        'use_hourly_statistics': (os.getenv('USE_HOURLY_STATISTICS') or 'true').lower() == 'true'
    }
