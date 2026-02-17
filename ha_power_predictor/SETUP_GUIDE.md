# HA Power Predictor - Setup & Visualization Guide

## Quick Start

### 1. Installation

1. In Home Assistant, navigate to **Settings → Add-ons → Add-on Store**
2. Click the three-dot menu in the top right
3. Select **Repositories**
4. Add the repository URL: `https://github.com/isaacjmannion/ha-power-predictor`
5. Find **HA Power Predictor** in the store
6. Click **Install**

### 2. Configuration

Before starting the add-on, configure your three data sources:

1. Go to the **Configuration** tab
2. Set your entity IDs:
   ```yaml
   power_entity: "sensor.sigen_plant_consumed_power"
   temperature_entity: "sensor.weather_temperature"
   forecast_entity: "weather.your_forecast_entity"
   ```
3. Adjust other settings as needed (see Configuration section below)
4. Click **Save**

### 3. Start the Add-on

1. Go to the **Info** tab
2. Enable **Start on boot** (optional)
3. Click **Start**
4. Wait for the add-on to initialise (check logs if needed)
5. Click **Open Web UI**

### 4. Work Through the Setup Steps

The web UI walks you through 4 steps to configure and run the model:

1. **Data Sources** — Review and confirm your power, temperature, and forecast entities
2. **Model Settings** — Configure history window, bin size, lag features, and quantile strategy
3. **Train & Evaluate** — Train the model and review accuracy metrics
4. **Predictions** — View the forecast results and publish sensors to Home Assistant

## Configuration Deep Dive

### Essential Settings

```yaml
# Your power consumption sensor - REQUIRED
power_entity: "sensor.sigen_plant_consumed_power"

# Historical temperature sensor - REQUIRED
temperature_entity: "sensor.weather_temperature"

# Weather forecast entity - REQUIRED
forecast_entity: "weather.your_forecast_entity"

# Days of history to use for training
# More days captures more seasonal patterns, but longer histories
# may include less relevant older data
history_days: 30

# Your timezone for proper time-of-day features
timezone: "Australia/Sydney"
```

### Model Tuning

```yaml
# Time bin size - how to group readings
# 60 = hourly average, 15 = 15-minute average
bin_size_minutes: 60

# How many previous power readings to use as features
# Higher values capture more patterns but increase training time
n_power_lags: 10

# How many previous temperature readings to use
n_temp_lags: 5

# Percentage of data to use for training vs testing
train_percentage: 80.0
```

### Quantile Strategy

**Option 1: Fixed Quantile (Simple)**
```yaml
quantile: 0.75  # 75th percentile
use_dynamic_quantile: false
```
Produces consistent conservative predictions. Around 75% of actual readings will fall at or below the forecast.

**Option 2: Dynamic Quantile (Recommended)**
```yaml
use_dynamic_quantile: true
peak_start: 9      # Peak starts at 9 AM
peak_end: 22       # Peak ends at 10 PM
peak_quantile: 0.75    # Conservative during peak hours
offpeak_quantile: 0.50 # Median estimate during off-peak
```
Applies a higher, more conservative quantile during peak hours and a more accurate median estimate overnight. Useful for solar optimisation, battery management, and time-of-use tariff planning.

## Understanding the Results

### Metrics Explained

**R² Score (Coefficient of Determination)**
- Range: -∞ to 1.0 (1.0 is perfect)
- What it means: How well the model explains variance in consumption
- Good: > 0.7 | Acceptable: 0.5–0.7 | Poor: < 0.5

**MAE (Mean Absolute Error)**
- Units: kW
- What it means: Average size of the prediction error
- Example: MAE of 0.5 kW means predictions are off by 500W on average

**RMSE (Root Mean Squared Error)**
- Units: kW
- What it means: Like MAE but penalises large errors more heavily
- If RMSE is much higher than MAE, there are occasional large prediction errors

**Coverage**
- Range: 0–100%
- What it means: Percentage of actual values that fall at or below the prediction
- Should roughly match your quantile setting (e.g. ~75% for q=0.75)
- Too low: Model is underestimating — consider increasing the quantile
- Too high: Model is overestimating — consider decreasing the quantile

### Published Sensors

After running a prediction, these sensors are created in Home Assistant:

| Sensor | Description |
|--------|-------------|
| `sensor.power_prediction_next_1h` | Average predicted load for the next hour |
| `sensor.power_prediction_next_6h` | Average predicted load for the next 6 hours |
| `sensor.power_prediction_next_12h` | Average predicted load for the next 12 hours |
| `sensor.power_prediction_next_24h` | Average predicted load for the next 24 hours |
| `sensor.power_prediction_next_48h` | Average predicted load for the next 48 hours |
| `sensor.power_prediction_full` | Complete dataset with all time-point predictions |

Each sensor includes attributes with individual time-point predictions that can be used in Lovelace dashboards or as trigger conditions in automations.

## Troubleshooting

### "Configuration Invalid" Error

Check that all entity IDs exist in your system:
```yaml
# Verify in Developer Tools → States that these exist:
sensor.sigen_plant_consumed_power
sensor.weather_temperature
weather.your_forecast_entity
```

### Low R² Score

Try these improvements:
1. Increase `history_days` to 60 or 90 to capture more usage patterns
2. Increase `n_power_lags` to give the model more context
3. Enable `use_dynamic_quantile` if your household has distinct peak/off-peak patterns
4. Verify the temperature sensor correlates meaningfully with your consumption

### Coverage Not Matching Quantile

**Coverage much lower than quantile (e.g. 60% when expecting 75%):**
The model is underestimating. Increase the quantile value, or increase both peak and off-peak quantiles in dynamic mode.

**Coverage much higher than quantile (e.g. 90% when expecting 75%):**
The model is overestimating. Decrease the quantile value for more realistic predictions.

### Predictions Not Appearing in Home Assistant

1. Check the add-on logs for API errors
2. Restart Home Assistant
3. Verify `homeassistant_api: true` is set in config.yaml
4. Check Developer Tools → States for sensors starting with `sensor.power_prediction_`

## Best Practices

1. **Run predictions regularly** — set up an automation to trigger predictions daily or on a schedule
2. **Monitor accuracy** — check that coverage aligns with your chosen quantile over time
3. **Adjust quantiles** — start at 0.75 and tune based on whether the model over- or under-estimates
4. **Match history to your needs** — 30 days is a good default; increase if your usage varies seasonally
5. **Match bin size to usage** — hourly bins (60 min) work well for most homes
6. **Combine with other data** — pair predictions with solar generation, battery state, and tariff sensors for richer automations

## Next Steps

1. Review prediction accuracy in the web UI after the first run
2. Tune quantile settings based on your coverage results
3. Use the published sensors in your Lovelace dashboard
4. Build automations that respond to upcoming high-load periods
5. Re-run predictions on a regular schedule to keep forecasts current

---

For more help, visit the [GitHub repository](https://github.com/isaacjmannion/ha-power-predictor) or the Home Assistant community forums.
