#!/usr/bin/env bash
set -e

echo "Starting HA Power Predictor..."

# Read configuration from options.json (Home Assistant passes config here)
if [ -f /data/options.json ]; then
    export POWER_ENTITY=$(jq -r '.power_entity // "sensor.power"' /data/options.json)
    export TEMPERATURE_ENTITY=$(jq -r '.temperature_entity // "sensor.temperature"' /data/options.json)
    export BIN_SIZE_MINUTES=$(jq -r '.bin_size_minutes // 60' /data/options.json)
    export N_POWER_LAGS=$(jq -r '.n_power_lags // 10' /data/options.json)
    export N_TEMP_LAGS=$(jq -r '.n_temp_lags // 5' /data/options.json)
    export TRAIN_PERCENTAGE=$(jq -r '.train_percentage // 80.0' /data/options.json)
    export QUANTILE=$(jq -r '.quantile // 0.75' /data/options.json)
    export USE_DYNAMIC_QUANTILE=$(jq -r '.use_dynamic_quantile // true' /data/options.json)
    export PEAK_START=$(jq -r '.peak_start // 9' /data/options.json)
    export PEAK_END=$(jq -r '.peak_end // 22' /data/options.json)
    export PEAK_QUANTILE=$(jq -r '.peak_quantile // 0.75' /data/options.json)
    export OFFPEAK_QUANTILE=$(jq -r '.offpeak_quantile // 0.50' /data/options.json)
    export HISTORY_DAYS=$(jq -r '.history_days // 90' /data/options.json)
    export TIMEZONE=$(jq -r '.timezone // "UTC"' /data/options.json)
    
    echo "Power Entity: ${POWER_ENTITY}"
    echo "Temperature Entity: ${TEMPERATURE_ENTITY}"
    echo "History Days: ${HISTORY_DAYS}"
else
    echo "WARNING: No options.json found, using defaults"
    export POWER_ENTITY="sensor.power"
    export TEMPERATURE_ENTITY="sensor.temperature"
    export BIN_SIZE_MINUTES=60
    export N_POWER_LAGS=10
    export N_TEMP_LAGS=5
    export TRAIN_PERCENTAGE=80.0
    export QUANTILE=0.75
    export USE_DYNAMIC_QUANTILE=true
    export PEAK_START=9
    export PEAK_END=22
    export PEAK_QUANTILE=0.75
    export OFFPEAK_QUANTILE=0.50
    export HISTORY_DAYS=90
    export TIMEZONE="UTC"
fi

# Get Home Assistant configuration
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"
export HA_URL="http://supervisor/core"

# Start the Flask application
cd /app
exec gunicorn --bind 0.0.0.0:8099 --workers 2 --timeout 300 app.main:app
