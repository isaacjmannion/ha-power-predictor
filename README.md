# Home Assistant Power Predictor Add-on Repository

[![GitHub Release](https://img.shields.io/github/release/isaacjmannion/ha-power-predictor.svg?style=flat-square)](https://github.com/isaacjmannion/ha-power-predictor/releases)
[![License](https://img.shields.io/github/license/isaacjmannion/ha-power-predictor.svg?style=flat-square)](LICENSE)

Forecast future power consumption directly within Home Assistant, using your historical usage and weather data.

## About

This repository contains the **HA Power Predictor** add-on for Home Assistant. It uses quantile regression to forecast future power consumption based on historical power usage, observed temperatures, and weather forecast data, with support for dynamic peak/off-peak modeling.

### Features

- 📊 **Configurable Predictions**: Quantile regression with adjustable percentiles to suit conservative or typical forecasts
- ⏰ **Peak/Off-Peak Modeling**: Apply different prediction strategies depending on the time of day
- 🖥️ **Web Interface**: Built-in UI with charts and model metrics
- 🔌 **Home Assistant Integration**: Prediction results are automatically published as HA sensors
- 📈 **Multiple Time Windows**: Predictions for 1h, 6h, 12h, 24h, and 48h ahead
- 🔄 **Multi-Step Forecasting**: Each step uses prior predicted values to build realistic forward projections
- 🏗️ **Multi-Architecture**: Supports all HA platforms (amd64, aarch64, armhf, etc.)

## Installation

### Via Add-on Store

1. Navigate to **Settings → Add-ons → Add-on Store** in Home Assistant
2. Click the menu (⋮) in the top right, then **Repositories**
3. Add this repository: `https://github.com/isaacjmannion/ha-power-predictor`
4. Find **HA Power Predictor** in the add-on store
5. Click **Install**

### Manual Installation

1. Navigate to the `/addons` folder of your Home Assistant instance
2. Clone this repository: `git clone https://github.com/isaacjmannion/ha-power-predictor`
3. Restart Home Assistant
4. Navigate to **Settings → Add-ons**
5. Find and install **HA Power Predictor**

## Quick Start

1. **Configure the add-on** with your three data sources:
   ```yaml
   power_entity: "sensor.your_power_sensor"
   temperature_entity: "sensor.your_historical_temperature_sensor"
   forecast_entity: "weather.your_forecast_entity"
   history_days: 30
   ```

2. **Start the add-on** and click **Open Web UI**

3. **Work through the 4 setup steps** in the web interface to configure and train the model

4. **View your predictions** in the web UI or via the published Home Assistant sensors

## Published Sensors

After running a prediction the following sensors are created in Home Assistant:

| Sensor | Description |
|--------|-------------|
| `sensor.power_prediction_next_1h` | Average predicted load for the next hour |
| `sensor.power_prediction_next_6h` | Average predicted load for the next 6 hours |
| `sensor.power_prediction_next_12h` | Average predicted load for the next 12 hours |
| `sensor.power_prediction_next_24h` | Average predicted load for the next 24 hours |
| `sensor.power_prediction_next_48h` | Average predicted load for the next 48 hours |
| `sensor.power_prediction_full` | Complete prediction dataset with all time-point values |

Each sensor includes attributes with individual time-point predictions that can be used in dashboards or automations.

## Configuration

See [ha_power_predictor/SETUP_GUIDE.md](ha_power_predictor/SETUP_GUIDE.md) for detailed configuration options.

Basic configuration:

| Option | Description | Default |
|--------|-------------|---------|
| `power_entity` | Your power consumption sensor | Required |
| `temperature_entity` | Historical temperature sensor | Required |
| `forecast_entity` | Weather forecast entity | Required |
| `history_days` | Days of history to use for training | 30 |
| `quantile` | Prediction percentile | 0.75 |
| `use_dynamic_quantile` | Enable peak/off-peak modes | true |

## Documentation

- **[README.md](ha_power_predictor/README.md)** - Basic add-on information
- **[DOCS.md](ha_power_predictor/DOCS.md)** - Comprehensive documentation
- **[SETUP_GUIDE.md](ha_power_predictor/SETUP_GUIDE.md)** - Complete setup guide with examples
- **[MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)** - Migration guide from desktop version

## Screenshots

### Web UI
![Web UI](https://via.placeholder.com/800x500/667eea/ffffff?text=Web+UI+Screenshot)

## Support

- **Issues**: [GitHub Issues](https://github.com/isaacjmannion/ha-power-predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/isaacjmannion/ha-power-predictor/discussions)
- **Community**: [Home Assistant Forum Thread](#)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Changelog

See [CHANGELOG.md](ha_power_predictor/CHANGELOG.md) for version history.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Repository Contents

```
ha-power-predictor/
├── ha_power_predictor/          # Main add-on directory
│   ├── config.yaml           # Add-on configuration
│   ├── Dockerfile            # Container definition
│   ├── run.sh               # Startup script
│   ├── requirements.txt     # Python dependencies
│   ├── README.md            # User documentation
│   ├── DOCS.md              # Detailed documentation
│   ├── SETUP_GUIDE.md       # Complete setup guide
│   ├── CHANGELOG.md         # Version history
│   ├── build.yaml           # Multi-arch build config
│   └── app/                 # Python application
│       ├── main.py          # Flask web server
│       ├── ha_client.py     # HA API client
│       ├── data_processing.py
│       ├── models.py
│       ├── config.py
│       └── templates/
│           └── index.html   # Web UI
├── repository.yaml           # Repository metadata
├── MIGRATION_SUMMARY.md      # Migration guide
└── README.md                # This file
```
