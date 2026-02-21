#!/usr/bin/env bash
set -e

echo "=== Debugging Configuration Loading ==="
echo "Checking /data/options.json..."

if [ ! -f /data/options.json ]; then
    echo "ERROR: /data/options.json does not exist!"
    exit 1
fi

echo "Contents of /data/options.json:"
cat /data/options.json
echo ""

# Create a temporary Python script to read config
cat > /tmp/read_config.py << 'PYEOF'
import json
import os

try:
    with open('/data/options.json') as f:
        config = json.load(f)
    
    print(f"DEBUG: Config loaded successfully, keys: {list(config.keys())}")
    
    # Helper to normalize booleans to lowercase strings
    def bool_str(val, default):
        if isinstance(val, bool):
            return 'true' if val else 'false'
        if val is None:
            return 'true' if default else 'false'
        return str(val).lower()
    
    # Write exports to a file instead of stdout
    with open('/tmp/env_exports.sh', 'w') as out:
        out.write(f"export POWER_ENTITY='{config.get('power_entity', 'sensor.power_consumption')}'\n")
        out.write(f"export TEMPERATURE_ENTITY='{config.get('temperature_entity', 'sensor.temperature')}'\n")
        out.write(f"export WEATHER_FORECAST_ENTITY='{config.get('weather_forecast_entity', 'weather.home')}'\n")
        out.write(f"export BIN_SIZE_MINUTES='{config.get('bin_size_minutes', 60)}'\n")
        out.write(f"export N_POWER_LAGS='{config.get('n_power_lags', 10)}'\n")
        out.write(f"export N_TEMP_LAGS='{config.get('n_temp_lags', 5)}'\n")
        out.write(f"export TRAIN_PERCENTAGE='{config.get('train_percentage', 80.0)}'\n")
        out.write(f"export QUANTILE='{config.get('quantile', 0.75)}'\n")
        out.write(f"export USE_DYNAMIC_QUANTILE='{bool_str(config.get('use_dynamic_quantile'), True)}'\n")
        out.write(f"export PEAK_START='{config.get('peak_start', 9)}'\n")
        out.write(f"export PEAK_END='{config.get('peak_end', 22)}'\n")
        out.write(f"export PEAK_QUANTILE='{config.get('peak_quantile', 0.75)}'\n")
        out.write(f"export OFFPEAK_QUANTILE='{config.get('offpeak_quantile', 0.50)}'\n")
        out.write(f"export HISTORY_DAYS='{config.get('history_days', 30)}'\n")
        out.write(f"export TIMEZONE='{config.get('timezone', 'UTC')}'\n")
        out.write(f"export USE_HOURLY_STATISTICS='{bool_str(config.get('use_hourly_statistics'), True)}'\n")
    
    print("DEBUG: Environment exports written to /tmp/env_exports.sh")
    
except Exception as e:
    print(f"ERROR reading config: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYEOF

# Run the Python script
echo "Running Python config parser..."
python3 /tmp/read_config.py

# Source the exported variables
if [ -f /tmp/env_exports.sh ]; then
    echo "Loading environment variables..."
    cat /tmp/env_exports.sh
    source /tmp/env_exports.sh
else
    echo "ERROR: /tmp/env_exports.sh was not created!"
    exit 1
fi

# Home Assistant API configuration (provided by Supervisor)
export HA_URL="${SUPERVISOR_URL:-http://supervisor/core}"
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"

echo ""
echo "=== Configuration Loaded ==="
echo "Power Entity: $POWER_ENTITY"
echo "Temperature Entity: $TEMPERATURE_ENTITY"
echo "Weather Forecast Entity: $WEATHER_FORECAST_ENTITY"
echo "History Days: $HISTORY_DAYS"
echo "Use Hourly Statistics: $USE_HOURLY_STATISTICS"
echo "Bin Size Minutes: $BIN_SIZE_MINUTES"
echo "================================"
echo ""

# Start the Flask application with gunicorn
echo "Starting gunicorn..."
exec gunicorn --bind 0.0.0.0:8099 \
    --workers 1 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app.main:app
