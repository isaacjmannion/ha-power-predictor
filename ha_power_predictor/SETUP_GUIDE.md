# HA Power Predictor - Setup & Visualization Guide

## Quick Start

### 1. Installation

1. In Home Assistant, navigate to **Settings → Add-ons → Add-on Store**
2. Click the three-dot menu in the top right
3. Select **Repositories**
4. Add your repository URL: `https://github.com/isaacmannion/ha-power-predictor`
5. Find "HA Power Predictor" in the store
6. Click **Install**

### 2. Configuration

Before starting the add-on, configure it:

1. Go to the **Configuration** tab
2. Set your entity IDs:
   ```yaml
   power_entity: "sensor.sigen_plant_consumed_power"
   temperature_entity: "sensor.weather_temperature"
   ```
3. Adjust other settings as needed (see Configuration section)
4. Click **Save**

### 3. Start the Add-on

1. Go to the **Info** tab
2. Enable "Start on boot" (optional)
3. Click **Start**
4. Wait for the add-on to start (check logs if needed)
5. Click **Open Web UI**

### 4. Run Your First Prediction

1. In the Web UI, review your configuration
2. Click **Run Prediction**
3. Wait for training to complete (30-60 seconds typically)
4. Review results and metrics

## Configuration Deep Dive

### Essential Settings

```yaml
# Your power consumption sensor - REQUIRED
power_entity: "sensor.sigen_plant_consumed_power"

# Temperature sensor that affects consumption - REQUIRED
temperature_entity: "sensor.weather_temperature"

# Days of history to use for training
# More days = better patterns, but slower training
history_days: 90

# Your timezone for proper time-of-day features
timezone: "Australia/Sydney"
```

### Model Tuning

```yaml
# Time bin size - how to group readings
# 60 = hourly average, 15 = 15-minute average
bin_size_minutes: 60

# How many previous power readings to use as features
# Higher = captures more patterns, but slower
n_power_lags: 10

# How many previous temperature readings to use
n_temp_lags: 5

# What % of data to use for training vs testing
train_percentage: 80.0
```

### Quantile Strategy

**Option 1: Fixed Quantile (Simple)**
```yaml
quantile: 0.75  # 75th percentile
use_dynamic_quantile: false
```
- Good for: Consistent conservative predictions
- Coverage: ~75% of actuals below prediction

**Option 2: Dynamic Quantile (Advanced)**
```yaml
use_dynamic_quantile: true
peak_start: 9      # Peak starts at 9 AM
peak_end: 22       # Peak ends at 10 PM
peak_quantile: 0.75    # Conservative during peak
offpeak_quantile: 0.50 # Median during off-peak
```
- Good for: Different strategies for different times
- Use cases: Solar optimization, TOU rates, battery management

## Understanding the Results

### Metrics Explained

**R² Score (Coefficient of Determination)**
- Range: -∞ to 1.0 (1.0 is perfect)
- What it means: How well the model explains variance
- Good: > 0.7
- Acceptable: 0.5 - 0.7
- Poor: < 0.5

**MAE (Mean Absolute Error)**
- Units: kW (same as your power)
- What it means: Average prediction error
- Example: MAE of 0.5 kW means predictions are off by 500W on average
- Lower is better

**RMSE (Root Mean Squared Error)**
- Units: kW
- What it means: Like MAE but penalizes large errors more
- If much higher than MAE: You have some large prediction errors
- Lower is better

**Coverage**
- Range: 0% to 100%
- What it means: % of actual values at or below prediction
- Should match your quantile (e.g., 75% for q=0.75)
- Too low: Model underestimates (dangerous for battery planning)
- Too high: Model overestimates (wasteful for battery charging)

### Published Sensors

After running a prediction, these sensors are created:

| Sensor | Description |
|--------|-------------|
| `sensor.power_prediction_next_1h` | Average predicted load for next hour |
| `sensor.power_prediction_next_6h` | Average for next 6 hours |
| `sensor.power_prediction_next_12h` | Average for next 12 hours |
| `sensor.power_prediction_next_24h` | Average for next 24 hours |
| `sensor.power_prediction_next_48h` | Average for next 48 hours |
| `sensor.power_prediction_full` | Complete dataset with all predictions |

Each sensor includes attributes with individual time-point predictions.

## Visualization Strategies

### Strategy 1: ApexCharts (Recommended)

**Installation:**
1. Install via HACS: Search for "ApexCharts Card"
2. Or manually: https://github.com/RomRider/apexcharts-card

**Basic Forecast Chart:**

```yaml
type: custom:apexcharts-card
header:
  show: true
  title: Power Forecast - Next 48 Hours
  show_states: true
  colorize_states: true
graph_span: 48h
now:
  show: true
  label: Now
series:
  - entity: sensor.sigen_plant_consumed_power
    name: Actual Consumption
    stroke_width: 2
    color: '#333333'
    show:
      legend_value: true
  - entity: sensor.power_prediction_full
    name: Predicted Consumption
    stroke_width: 2
    color: '#667eea'
    type: line
    show:
      legend_value: true
    data_generator: |
      return entity.attributes.predictions.slice(0, 48).map((p) => {
        return [new Date(p.time).getTime(), p.predicted];
      });
```

**Advanced Chart with Confidence Band:**

```yaml
type: custom:apexcharts-card
header:
  show: true
  title: Power Forecast with Uncertainty
graph_span: 48h
series:
  - entity: sensor.sigen_plant_consumed_power
    name: Actual
    stroke_width: 3
    color: black
  - entity: sensor.power_prediction_full
    name: Predicted
    stroke_width: 2
    color: '#667eea'
    data_generator: |
      return entity.attributes.predictions.slice(0, 48).map((p) => {
        return [new Date(p.time).getTime(), p.predicted];
      });
  # Upper bound (+10%)
  - entity: sensor.power_prediction_full
    name: Upper Bound
    stroke_width: 1
    color: '#667eea'
    opacity: 0.3
    data_generator: |
      return entity.attributes.predictions.slice(0, 48).map((p) => {
        return [new Date(p.time).getTime(), p.predicted * 1.1];
      });
  # Lower bound (-10%)
  - entity: sensor.power_prediction_full
    name: Lower Bound
    stroke_width: 1
    color: '#667eea'
    opacity: 0.3
    data_generator: |
      return entity.attributes.predictions.slice(0, 48).map((p) => {
        return [new Date(p.time).getTime(), p.predicted * 0.9];
      });
```

### Strategy 2: Gauge Cards for Quick Overview

```yaml
type: vertical-stack
cards:
  - type: gauge
    entity: sensor.power_prediction_next_1h
    name: Next Hour
    min: 0
    max: 10
    severity:
      green: 0
      yellow: 5
      red: 7.5
  - type: gauge
    entity: sensor.power_prediction_next_6h
    name: Next 6 Hours
    min: 0
    max: 10
  - type: gauge
    entity: sensor.power_prediction_next_24h
    name: Next 24 Hours
    min: 0
    max: 10
```

### Strategy 3: Entity Cards with Statistics

```yaml
type: entities
title: Energy Predictions
entities:
  - type: attribute
    entity: sensor.power_prediction_next_1h
    attribute: average
    name: Next 1h Average
    icon: mdi:clock-outline
  - type: attribute
    entity: sensor.power_prediction_next_1h
    attribute: maximum
    name: Next 1h Peak
    icon: mdi:arrow-up-bold
  - type: attribute
    entity: sensor.power_prediction_next_24h
    attribute: average
    name: Next 24h Average
    icon: mdi:calendar-clock
  - type: attribute
    entity: sensor.power_prediction_next_24h
    attribute: maximum
    name: Next 24h Peak
    icon: mdi:flash-alert
```

### Strategy 4: Combined Dashboard

Create a comprehensive dashboard:

```yaml
title: Power Forecasting
views:
  - title: Overview
    cards:
      # Prediction vs Actual Chart
      - type: custom:apexcharts-card
        header:
          show: true
          title: 48-Hour Forecast
        graph_span: 48h
        series:
          - entity: sensor.sigen_plant_consumed_power
            name: Actual
            stroke_width: 2
          - entity: sensor.power_prediction_full
            name: Predicted
            stroke_width: 2
            data_generator: |
              return entity.attributes.predictions.slice(0, 48).map((p) => {
                return [new Date(p.time).getTime(), p.predicted];
              });
      
      # Quick Stats
      - type: horizontal-stack
        cards:
          - type: statistic
            entity: sensor.power_prediction_next_1h
            name: Next Hour
            icon: mdi:clock-outline
          - type: statistic
            entity: sensor.power_prediction_next_6h
            name: Next 6 Hours
            icon: mdi:clock-time-six-outline
          - type: statistic
            entity: sensor.power_prediction_next_24h
            name: Next Day
            icon: mdi:calendar-today
      
      # Detailed Predictions
      - type: markdown
        title: Upcoming Peaks
        content: |
          {% set preds = state_attr('sensor.power_prediction_full', 'predictions')[:24] %}
          {% set max_pred = preds | map(attribute='predicted') | max %}
          {% set max_time = (preds | selectattr('predicted', 'equalto', max_pred) | first).time %}
          **Peak in next 24h:** {{ max_pred | round(2) }} kW at {{ max_time | as_timestamp | timestamp_custom('%H:%M') }}
```

## Automation Examples

### Example 1: High Load Warning

```yaml
automation:
  - alias: "Energy: High Load Warning"
    trigger:
      - platform: numeric_state
        entity_id: sensor.power_prediction_next_1h
        above: 5.0
    condition:
      - condition: time
        after: "06:00:00"
        before: "22:00:00"
    action:
      - service: notify.mobile_app
        data:
          title: "⚡ High Energy Load Predicted"
          message: >
            Predicted consumption in next hour: {{ states('sensor.power_prediction_next_1h') }} kW
            Consider delaying high-power appliances.
```

### Example 2: Battery Discharge Planning

```yaml
automation:
  - alias: "Battery: Optimize Discharge"
    trigger:
      - platform: time_pattern
        hours: "/1"  # Every hour
    condition:
      - condition: numeric_state
        entity_id: sensor.battery_level
        above: 50
    action:
      - service: number.set_value
        target:
          entity_id: number.battery_discharge_power
        data:
          value: >
            {% set pred = states('sensor.power_prediction_next_1h') | float %}
            {% set solar = states('sensor.solar_power') | float %}
            {% set needed = pred - solar %}
            {{ [0, needed, 5.0] | sort | list | nth(1) }}
```

### Example 3: Pre-cool House

```yaml
automation:
  - alias: "Climate: Pre-cool Before Peak"
    trigger:
      - platform: numeric_state
        entity_id: sensor.power_prediction_next_6h
        above: 6.0
    condition:
      - condition: numeric_state
        entity_id: sensor.temperature
        above: 24
      - condition: state
        entity_id: climate.living_room
        state: "off"
    action:
      - service: climate.set_temperature
        target:
          entity_id: climate.living_room
        data:
          temperature: 22
          hvac_mode: cool
      - service: notify.mobile_app
        data:
          message: "Pre-cooling house before predicted high load period"
```

## Troubleshooting

### "Configuration Invalid" Error

Check that entity IDs exist:
```yaml
# Verify in Developer Tools → States that these exist:
sensor.sigen_plant_consumed_power
sensor.weather_temperature
```

### Low R² Score

Try these improvements:
1. Increase `history_days` to 90 or more
2. Increase `n_power_lags` to capture more patterns
3. Enable `use_dynamic_quantile` if you have distinct peak/off-peak patterns
4. Verify temperature sensor actually correlates with usage

### Coverage Not Matching Quantile

**Coverage much lower than quantile (e.g., 60% when expecting 75%):**
- Model is underestimating
- Increase quantile value
- For dynamic mode, increase both peak and off-peak quantiles

**Coverage much higher than quantile (e.g., 90% when expecting 75%):**
- Model is overestimating
- Decrease quantile value
- Consider using a lower quantile for more accurate predictions

### Predictions Not Appearing in HA

1. Check add-on logs for API errors
2. Restart Home Assistant
3. Verify `homeassistant_api: true` in config.yaml
4. Check Developer Tools → States for sensors starting with `sensor.power_prediction_`

## Best Practices

1. **Run predictions regularly**: Set up an automation to run predictions daily
2. **Monitor accuracy**: Track how predictions compare to actuals over time
3. **Adjust quantiles**: Start conservative (0.75) and adjust based on coverage
4. **Use appropriate history**: 30-90 days for most use cases
5. **Match bin size to usage**: Hourly bins work well for most homes
6. **Visualize predictions**: Use ApexCharts for best insight into patterns
7. **Combine with other sensors**: Use predictions with solar, battery, and tariff data

## Next Steps

1. Install ApexCharts card for better visualization
2. Create a dedicated dashboard for energy forecasting
3. Set up automations to act on predictions
4. Monitor and tune quantile settings
5. Consider adding more lag features if accuracy is low
6. Integrate with battery management or HVAC systems

---

For more help, visit the GitHub repository or Home Assistant community forums.
