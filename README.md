# Home Assistant HA Power Predictor Add-on Repository

[![GitHub Release](https://img.shields.io/github/release/yourusername/ha-power-predictor.svg?style=flat-square)](https://github.com/yourusername/ha-power-predictor/releases)
[![License](https://img.shields.io/github/license/yourusername/ha-power-predictor.svg?style=flat-square)](LICENSE)

Predict future power consumption using machine learning, directly integrated with Home Assistant.

![Energy Predictor](https://via.placeholder.com/800x400/667eea/ffffff?text=Energy+Load+Predictor)

## About

This repository contains the **HA Power Predictor** add-on for Home Assistant. It uses quantile regression to forecast future power consumption based on historical power usage and temperature data, with support for dynamic peak/off-peak modeling.

### Features

- 🔮 **Accurate Predictions**: Quantile regression with configurable percentiles
- ⏰ **Dynamic Modeling**: Different prediction strategies for peak vs off-peak hours
- 📊 **Rich Visualization**: Beautiful web UI with charts and metrics
- 🔌 **Full Integration**: Automatic sensor publishing to Home Assistant
- 🚀 **Easy to Use**: One-click predictions via web interface
- 📈 **Multiple Windows**: Get predictions for 1h, 6h, 12h, 24h, and 48h ahead
- 🎯 **Iterative Forecasting**: Realistic multi-step predictions using predicted lags
- 🏗️ **Multi-Architecture**: Supports all HA platforms (amd64, aarch64, armhf, etc.)

## Installation

### Via Add-on Store

1. Navigate to **Settings → Add-ons → Add-on Store** in Home Assistant
2. Click the menu (⋮) in the top right, then **Repositories**
3. Add this repository: `https://github.com/yourusername/ha-power-predictor`
4. Find **HA Power Predictor** in the add-on store
5. Click **Install**

### Manual Installation

1. Navigate to the `/addons` folder of your Home Assistant instance
2. Clone this repository: `git clone https://github.com/yourusername/ha-power-predictor`
3. Restart Home Assistant
4. Navigate to **Settings → Add-ons**
5. Find and install **HA Power Predictor**

## Quick Start

1. **Configure the add-on:**
   ```yaml
   power_entity: "sensor.your_power_sensor"
   temperature_entity: "sensor.your_temperature_sensor"
   history_days: 90
   ```

2. **Start the add-on** and click "Open Web UI"

3. **Click "Run Prediction"** and wait for results

4. **View your predictions** in:
   - The web UI with interactive charts
   - Home Assistant sensors (`sensor.power_prediction_*`)
   - Your dashboards using ApexCharts or other cards

## Configuration

See [ha_power_predictor/README.md](ha_power_predictor/README.md) for detailed configuration options.

Basic configuration:

| Option | Description | Default |
|--------|-------------|---------|
| `power_entity` | Your power consumption sensor | Required |
| `temperature_entity` | Your temperature sensor | Required |
| `history_days` | Days of history to use | 90 |
| `quantile` | Prediction percentile | 0.75 |
| `use_dynamic_quantile` | Enable peak/off-peak modes | true |

## Documentation

- **[README.md](ha_power_predictor/README.md)** - Basic add-on information
- **[DOCS.md](ha_power_predictor/DOCS.md)** - Comprehensive documentation
- **[SETUP_GUIDE.md](ha_power_predictor/SETUP_GUIDE.md)** - Complete setup guide with examples
- **[MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)** - Migration guide from desktop version

## Visualization Examples

### ApexCharts (Recommended)

```yaml
type: custom:apexcharts-card
header:
  show: true
  title: Power Forecast - Next 48 Hours
graph_span: 48h
series:
  - entity: sensor.power_consumption
    name: Actual
    stroke_width: 2
  - entity: sensor.power_prediction_full
    name: Predicted
    stroke_width: 2
    data_generator: |
      return entity.attributes.predictions.slice(0, 48).map((p) => {
        return [new Date(p.time).getTime(), p.predicted];
      });
```

See [SETUP_GUIDE.md](ha_power_predictor/SETUP_GUIDE.md) for more visualization examples.

## Use Cases

- **Battery Management**: Size discharge/charge based on predicted consumption
- **Solar Optimization**: Plan self-consumption vs grid export
- **Load Shifting**: Move consumption to predicted low-use periods
- **TOU Rate Optimization**: Avoid high-cost periods
- **HVAC Pre-conditioning**: Cool/heat before predicted peaks
- **Capacity Planning**: Understand typical usage patterns

## Automation Example

```yaml
automation:
  - alias: "Battery: Discharge During Peak"
    trigger:
      - platform: numeric_state
        entity_id: sensor.power_prediction_next_1h
        above: 5.0
    action:
      - service: number.set_value
        target:
          entity_id: number.battery_discharge_power
        data:
          value: 5.0
```

## Screenshots

### Web UI
![Web UI](https://via.placeholder.com/800x500/667eea/ffffff?text=Web+UI+Screenshot)

### Home Assistant Dashboard
![Dashboard](https://via.placeholder.com/800x500/667eea/ffffff?text=Dashboard+Screenshot)

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ha-power-predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ha-power-predictor/discussions)
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

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/)
- Web UI powered by [Flask](https://flask.palletsprojects.com/) and [Chart.js](https://www.chartjs.org/)
- Inspired by the Home Assistant community

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

---

**Made with ❤️ for the Home Assistant community**
