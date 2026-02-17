# HA Power Predictor Documentation

## Overview

The HA Power Predictor add-on uses quantile regression to forecast future power consumption based on historical power usage and temperature data from your Home Assistant instance.

## How It Works

### Data Collection

The add-on fetches historical data from Home Assistant's recorder database using the REST API:

1. **Power Consumption**: Historical readings from your configured power sensor
2. **Temperature**: Historical temperature readings that influence energy usage

### Data Processing

Raw sensor data is processed into time-binned averages:

- Configurable bin sizes (15 min to 24 hours)
- Temporal features extracted: year, month, day of week, hour, minute
- Lagged features: Previous power and temperature readings
- Missing data handled automatically

### Model Training

The add-on uses Quantile Regression from scikit-learn:

- **Standard Mode**: Single quantile for all predictions
- **Dynamic Mode**: Different quantiles for peak vs off-peak hours
  - Peak hours: Higher quantile for conservative estimates
  - Off-peak hours: Lower quantile for more accurate predictions

### Prediction

Predictions are made iteratively:

1. For the first time step, use actual historical lag values
2. For subsequent time steps, use previous predictions as lag values
3. This simulates real-time forecasting where future actuals aren't known

### Publishing to Home Assistant

Predictions are published as sensors with rich attributes:

- Individual time-point predictions
- Aggregated statistics (average, maximum)
- Metadata (source entity, time window)

## Configuration Guide

### Recommended Settings for Different Use Cases

#### Conservative Battery Planning
```yaml
quantile: 0.85
use_dynamic_quantile: true
peak_quantile: 0.90
offpeak_quantile: 0.75
```

This ensures you rarely underestimate consumption, important for battery sizing.

#### Accurate Forecasting
```yaml
quantile: 0.50
use_dynamic_quantile: false
```

This gives median predictions, balanced between over and under estimation.

#### Solar Optimization
```yaml
use_dynamic_quantile: true
peak_start: 9
peak_end: 17
peak_quantile: 0.75
offpeak_quantile: 0.60
```

Higher quantiles during solar production hours helps optimize self-consumption.

### Lag Features

Lag features use previous power/temperature readings as predictors:

- **Power Lags**: Captures usage patterns and autocorrelation
  - Recommended: 10-24 for hourly bins
  - More lags = better pattern recognition but slower training
  
- **Temperature Lags**: Captures delayed thermal effects
  - Recommended: 5-12 for hourly bins
  - Useful for heating/cooling prediction

### Historical Data

- **Minimum**: 7 days for basic patterns
- **Recommended**: 30-90 days for seasonal patterns
- **Maximum**: 365 days for full year coverage

More data improves accuracy but increases computation time.

## Interpreting Results

### Metrics

- **R² Score**: How well predictions fit actual data (higher is better, max 1.0)
  - > 0.7: Excellent
  - 0.5-0.7: Good
  - < 0.5: Poor (consider adjusting configuration)

- **MAE** (Mean Absolute Error): Average prediction error in kW
  - Lower is better
  - Compare to your typical consumption range

- **RMSE** (Root Mean Squared Error): Prediction error emphasizing large mistakes
  - Lower is better
  - Higher than MAE indicates some large errors

- **Coverage**: Percentage of actual values at or below prediction
  - Should match your quantile (e.g., 75% for 0.75 quantile)
  - Much lower = model underestimating
  - Much higher = model overestimating

### Using Predictions

The add-on creates sensors you can use in:

1. **Automations**: Trigger actions based on predicted consumption
2. **Energy Dashboard**: Compare predictions to actuals
3. **Battery Management**: Size discharge based on upcoming usage
4. **Load Shifting**: Move consumption to predicted low-use periods

## Visualization Options

### Option 1: ApexCharts Card

```yaml
type: custom:apexcharts-card
header:
  show: true
  title: 48-Hour Power Forecast
  show_states: true
graph_span: 48h
series:
  - entity: sensor.power_consumption
    name: Actual
    stroke_width: 2
    color: '#333'
  - entity: sensor.power_prediction_full
    name: Predicted
    stroke_width: 2
    color: '#667eea'
    data_generator: |
      return entity.attributes.predictions.slice(0, 48).map((p) => {
        return [new Date(p.time).getTime(), p.predicted];
      });
```

### Option 2: Mini Graph Card

```yaml
type: custom:mini-graph-card
entities:
  - entity: sensor.power_prediction_next_1h
    name: Next 1h
  - entity: sensor.power_prediction_next_6h
    name: Next 6h
  - entity: sensor.power_prediction_next_24h
    name: Next 24h
hours_to_show: 48
points_per_hour: 1
```

### Option 3: Entities Card

```yaml
type: entities
title: Energy Predictions
entities:
  - entity: sensor.power_prediction_next_1h
    name: Next Hour Average
  - entity: sensor.power_prediction_next_6h
    name: Next 6 Hours Average
  - entity: sensor.power_prediction_next_12h
    name: Next 12 Hours Average
  - entity: sensor.power_prediction_next_24h
    name: Next 24 Hours Average
```

## Troubleshooting

### "No historical data found"

- Verify your entity IDs are correct
- Check that recorder is enabled and storing history
- Ensure entities have been active for at least a few hours
- Try increasing `history_days`

### "Insufficient data"

- Need at least 100 time bins after processing
- Reduce `bin_size_minutes`
- Increase `history_days`
- Check for gaps in sensor data

### Poor Prediction Accuracy

- Try different quantile values
- Enable dynamic quantile mode
- Increase number of lag features
- Use more historical data
- Check if temperature sensor is relevant to your consumption

### Predictions Not Publishing

- Check Home Assistant logs for API errors
- Verify add-on has `homeassistant_api: true` in config
- Restart Home Assistant after first run

## Advanced Usage

### Automation Example

```yaml
automation:
  - alias: "High Load Warning"
    trigger:
      - platform: numeric_state
        entity_id: sensor.power_prediction_next_1h
        above: 5.0
    action:
      - service: notify.mobile_app
        data:
          message: "High energy load predicted in next hour: {{ states('sensor.power_prediction_next_1h') }} kW"
```

### Template Sensor Example

```yaml
template:
  - sensor:
      - name: "Prediction Accuracy"
        unit_of_measurement: "%"
        state: >
          {% set pred = states('sensor.power_prediction_next_1h') | float %}
          {% set actual = states('sensor.power_consumption') | float %}
          {% if pred > 0 %}
            {{ (100 - (abs(pred - actual) / pred * 100)) | round(1) }}
          {% else %}
            0
          {% endif %}
```

## Performance Considerations

- Training time scales with data size and lag features
- Typical training on 90 days of hourly data: 30-60 seconds
- Web UI may timeout on very large datasets (>365 days)
- Consider running predictions during off-peak hours via automation

## Privacy & Security

- All data processing happens locally in your Home Assistant instance
- No external API calls or data sharing
- Predictions are stored only in Home Assistant's database
- Supervisor token is never logged or exposed
