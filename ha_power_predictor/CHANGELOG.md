# Changelog

All notable changes to this add-on will be documented in this file.

## [0.1.0] - 2026-02-14

### Added
- Initial release of HA Power Predictor add-on
- Quantile regression model for energy forecasting
- Dynamic peak/off-peak quantile support
- Web UI for configuration and results visualization
- Automatic sensor publishing to Home Assistant
- Support for multiple prediction windows (1h, 6h, 12h, 24h, 48h)
- Iterative forecasting using predicted lag values
- Configurable temporal and lagged features
- Historical data fetching from Home Assistant API
- Multi-architecture support (aarch64, amd64, armhf, armv7, i386)

### Features
- Real-time prediction chart
- Performance metrics display (R², MAE, RMSE, Coverage)
- Detailed prediction table for next 48 hours
- Responsive web interface
- Comprehensive configuration options
- ApexCharts integration examples

## [0.1.0] - 2026-02-16

### Added
- Bug fixed to run as HA addon