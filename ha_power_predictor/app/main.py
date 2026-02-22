"""
Main Flask application for HA Power Predictor add-on.
Provides web UI and API endpoints for power prediction.
"""

from flask import Flask, render_template, jsonify, request
import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz
import traceback

from .ha_client import HomeAssistantClient
from .data_processing import process_ha_data, process_ha_statistics, add_lagged_features, get_default_features
from .models import QuantileRegressionModel, predict_iterative
from .config import get_config_from_env

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Global state to store fetched data
app_state = {
    'power_data': None,
    'weather_history': None,
    'weather_forecast': None,
    'prediction_results': None,
    'last_update': {}
}


@app.route('/')
def index():
    """Render main page."""
    config = get_config_from_env()
    return render_template('index.html', config=config)


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    config = get_config_from_env()
    return jsonify(config)


@app.route('/api/fetch-power', methods=['POST'])
def fetch_power():
    """Fetch historical power consumption data from Home Assistant."""
    try:
        config = get_config_from_env()
        use_stats = config.get('use_hourly_statistics', False)

        app.logger.info(f"=== FETCH POWER DEBUG ===")
        app.logger.info(f"use_hourly_statistics config value: {use_stats}")
        app.logger.info(f"USE_HOURLY_STATISTICS env var: {os.getenv('USE_HOURLY_STATISTICS')}")

        ha_client = HomeAssistantClient(
            base_url=os.getenv('HA_URL', 'http://supervisor/core'),
            token=os.getenv('SUPERVISOR_TOKEN')
        )

        app.logger.info(f"Fetching {config['history_days']} days of power data...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=config['history_days'])

        if use_stats:
            app.logger.info("✓ Using hourly statistics API (should fetch ~720 records for 30 days)...")
            power_data = ha_client.get_statistics(config['power_entity'], start_time, end_time, 'hour')
        else:
            app.logger.info("✗ Using raw history API (will fetch many thousands of records)...")
            power_data = ha_client.get_history(config['power_entity'], start_time, end_time)

        if not power_data:
            return jsonify({'error': 'No power data found'}), 400

        app_state['power_data'] = power_data
        app_state['last_update']['power'] = datetime.now().isoformat()

        # Extract values for statistics (different structure for stats vs history)
        values = []
        if use_stats:
            for record in power_data:
                try:
                    mean_value = record.get('mean')
                    if mean_value is not None:
                        values.append(float(mean_value))
                except (ValueError, TypeError, KeyError):
                    continue
        else:
            for record in power_data:
                try:
                    state_str = str(record.get('state', ''))
                    if state_str.lower() not in ('unknown', 'unavailable', 'none', ''):
                        values.append(float(state_str))
                except (ValueError, TypeError, AttributeError):
                    continue

        if not values:
            return jsonify({'error': 'No valid power values found in data'}), 400

        stats = {
            'count': len(power_data),
            'valid_count': len(values),
            'min': round(min(values), 2),
            'max': round(max(values), 2),
            'mean': round(sum(values) / len(values), 2),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'source': 'hourly_statistics' if use_stats else 'raw_history'
        }

        # Sample records (different structure)
        sample = []
        if use_stats:
            for record in power_data[:10]:
                try:
                    sample.append({
                        'timestamp': str(record.get('start', 'N/A')),
                        'mean': str(record.get('mean', 'N/A'))
                    })
                except Exception:
                    continue
        else:
            for record in power_data[:10]:
                try:
                    sample.append({
                        'timestamp': str(record.get('last_changed', record.get('last_updated', 'N/A'))),
                        'state': str(record.get('state', 'N/A'))
                    })
                except Exception:
                    continue

        return jsonify({
            'success': True,
            'message': f'Fetched {len(power_data)} power records ({len(values)} valid)',
            'statistics': stats,
            'sample': sample
        })

    except Exception as e:
        app.logger.error(f"Error fetching power data: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to fetch power data: {str(e)}'}), 500


@app.route('/api/fetch-weather-history', methods=['POST'])
def fetch_weather_history():
    """Fetch historical weather/temperature data from Home Assistant."""
    try:
        config = get_config_from_env()
        use_stats = config.get('use_hourly_statistics', False)

        app.logger.info(f"=== FETCH TEMPERATURE DEBUG ===")
        app.logger.info(f"use_hourly_statistics config value: {use_stats}")

        temp_entity = config.get('temperature_entity')
        if not temp_entity:
            return jsonify({'error': 'temperature_entity not configured.'}), 400

        ha_client = HomeAssistantClient(
            base_url=os.getenv('HA_URL', 'http://supervisor/core'),
            token=os.getenv('SUPERVISOR_TOKEN')
        )

        app.logger.info(f"Fetching {config['history_days']} days of temperature history from {temp_entity}...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=config['history_days'])

        if use_stats:
            app.logger.info("✓ Using hourly statistics API for temperature...")
            temp_data = ha_client.get_statistics(temp_entity, start_time, end_time, 'hour')
        else:
            app.logger.info("✗ Using raw history API for temperature...")
            temp_data = ha_client.get_history(temp_entity, start_time, end_time)

        if not temp_data:
            return jsonify({'error': f'No temperature history found for {temp_entity}'}), 400

        app_state['weather_history'] = temp_data
        app_state['last_update']['weather_history'] = datetime.now().isoformat()

        # Extract values (different structure for stats vs history)
        values = []
        if use_stats:
            for record in temp_data:
                try:
                    mean_value = record.get('mean')
                    if mean_value is not None:
                        values.append(float(mean_value))
                except (ValueError, TypeError, KeyError):
                    continue
        else:
            for record in temp_data:
                try:
                    state_str = str(record.get('state', ''))
                    if state_str.lower() not in ('unknown', 'unavailable', 'none', ''):
                        values.append(float(state_str))
                except (ValueError, TypeError, AttributeError):
                    continue

        if not values:
            return jsonify({'error': 'No valid temperature values found in data'}), 400

        stats = {
            'count': len(temp_data),
            'valid_count': len(values),
            'min': round(min(values), 1),
            'max': round(max(values), 1),
            'mean': round(sum(values) / len(values), 1),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'entity': temp_entity,
            'source': 'hourly_statistics' if use_stats else 'raw_history'
        }

        # Sample records (different structure)
        sample = []
        if use_stats:
            for record in temp_data[:10]:
                try:
                    sample.append({
                        'timestamp': str(record.get('start', 'N/A')),
                        'mean': str(record.get('mean', 'N/A'))
                    })
                except Exception:
                    continue
        else:
            for record in temp_data[:10]:
                try:
                    sample.append({
                        'timestamp': str(record.get('last_changed', record.get('last_updated', 'N/A'))),
                        'state': str(record.get('state', 'N/A'))
                    })
                except Exception:
                    continue

        return jsonify({
            'success': True,
            'message': f'Fetched {len(temp_data)} temperature records ({len(values)} valid) from {temp_entity}',
            'statistics': stats,
            'sample': sample
        })

    except Exception as e:
        app.logger.error(f"Error fetching weather history: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to fetch weather history: {str(e)}'}), 500


@app.route('/api/fetch-weather-forecast', methods=['POST'])
def fetch_weather_forecast():
    """Fetch future weather forecast from Home Assistant."""
    try:
        config = get_config_from_env()
        weather_entity = config.get('weather_forecast_entity', 'weather.home')

        ha_client = HomeAssistantClient(
            base_url=os.getenv('HA_URL', 'http://supervisor/core'),
            token=os.getenv('SUPERVISOR_TOKEN')
        )

        forecasts = ha_client.get_weather_forecast(weather_entity, 'hourly')

        if not forecasts:
            return jsonify({'error': f'No weather forecast found for {weather_entity}.'}), 400

        app_state['weather_forecast'] = forecasts
        app_state['last_update']['weather_forecast'] = datetime.now().isoformat()

        temps = []
        for f in forecasts:
            if 'temperature' in f:
                try:
                    temps.append(float(f['temperature']))
                except (ValueError, TypeError):
                    continue

        if not temps:
            return jsonify({'error': 'No valid temperature values found in forecast'}), 400

        stats = {
            'count': len(forecasts),
            'hours_ahead': len(forecasts),
            'min_temp': round(min(temps), 1),
            'max_temp': round(max(temps), 1),
            'mean_temp': round(sum(temps) / len(temps), 1),
            'entity': weather_entity
        }

        sample = []
        for f in forecasts[:10]:
            try:
                sample.append({
                    'datetime': str(f.get('datetime', 'N/A')),
                    'temperature': str(f.get('temperature', 'N/A')),
                    'condition': str(f.get('condition', 'N/A'))
                })
            except Exception:
                continue

        return jsonify({
            'success': True,
            'message': f'Fetched {len(forecasts)} hours of forecast from {weather_entity}',
            'statistics': stats,
            'sample': sample
        })

    except Exception as e:
        app.logger.error(f"Error fetching weather forecast: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to fetch weather forecast: {str(e)}'}), 500


def _build_future_features(config, df_history, features, forecast_data):
    """
    Build a feature matrix for future time bins using weather forecast temperatures.

    Returns a DataFrame of future rows ready for prediction.
    """
    tz = pytz.timezone(config['timezone'])
    bin_minutes = config['bin_size_minutes']
    now = datetime.now(tz)

    # Round now up to the next bin boundary
    bin_td = timedelta(minutes=bin_minutes)
    seconds_since_epoch = now.timestamp()
    bin_seconds = bin_minutes * 60
    next_bin_ts = (int(seconds_since_epoch / bin_seconds) + 1) * bin_seconds
    next_bin = datetime.fromtimestamp(next_bin_ts, tz=tz)

    # How many future bins do we need? Use forecast length or 48h, whichever is smaller
    n_bins = min(48, len(forecast_data)) if forecast_data else 48
    future_timestamps = [next_bin + timedelta(minutes=bin_minutes * i) for i in range(n_bins)]

    app.logger.info(f"Building future features: {n_bins} bins from {next_bin} in {config['timezone']}")

    # Build forecast temperature lookup: datetime string -> temperature
    # Forecast datetimes are hourly; interpolate to bin resolution
    forecast_temps = {}
    for f in forecast_data:
        try:
            dt = pd.to_datetime(f['datetime']).tz_convert(tz)
            forecast_temps[dt] = float(f['temperature'])
        except Exception:
            continue

    def get_forecast_temp(ts):
        """Get nearest forecast temperature for a given timestamp."""
        if not forecast_temps:
            # Fall back to mean historical temperature
            return float(df_history['temperature'].mean())
        # Find the closest forecast hour
        closest = min(forecast_temps.keys(), key=lambda ft: abs((ft - ts).total_seconds()))
        return forecast_temps[closest]

    # Build future rows
    future_rows = []
    for ts in future_timestamps:
        future_rows.append({
            'timestamp': ts,
            'year': ts.year,
            'month': ts.month,
            'day_of_week': ts.weekday(),
            'hour': ts.hour,
            'minute': ts.minute,
            'temperature': get_forecast_temp(ts),
            'consumption': np.nan  # unknown future
        })

    df_future = pd.DataFrame(future_rows)

    # Add power lag features using last known actual values from history
    n_power_lags = config['n_power_lags']
    n_temp_lags = config['n_temp_lags']

    if n_power_lags > 0:
        last_power = df_history['consumption'].iloc[-n_power_lags:].values[::-1]
        for i in range(1, n_power_lags + 1):
            df_future[f'power_lag_{i}'] = last_power[i - 1] if i <= len(last_power) else df_history['consumption'].mean()

    if n_temp_lags > 0:
        last_temp = df_history['temperature'].iloc[-n_temp_lags:].values[::-1]
        for i in range(1, n_temp_lags + 1):
            df_future[f'temp_lag_{i}'] = last_temp[i - 1] if i <= len(last_temp) else df_history['temperature'].mean()

    return df_future


@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train the prediction model on all historical data, then predict forward from now.
    Also evaluates model quality using a train/test split on historical data.
    """
    try:
        if not app_state['power_data']:
            return jsonify({'error': 'Please fetch power data first'}), 400
        if not app_state['weather_history']:
            return jsonify({'error': 'Please fetch weather history first'}), 400

        config = get_config_from_env()
        use_stats = config.get('use_hourly_statistics', False)

        # --- Process historical data ---
        app.logger.info("Processing historical data...")
        if use_stats:
            # Use statistics processing (already hourly)
            df = process_ha_statistics(
                app_state['power_data'],
                app_state['weather_history'],
                timezone=config['timezone']
            )
        else:
            # Use raw history processing with binning
            df = process_ha_data(
                app_state['power_data'],
                app_state['weather_history'],
                bin_size_minutes=config['bin_size_minutes'],
                timezone=config['timezone']
            )

        if len(df) < 100:
            return jsonify({'error': f'Insufficient data: only {len(df)} records'}), 400

        df = add_lagged_features(df, n_power_lags=config['n_power_lags'], n_temp_lags=config['n_temp_lags'])

        features = get_default_features()
        for i in range(1, config['n_power_lags'] + 1):
            features.append(f'power_lag_{i}')
        for i in range(1, config['n_temp_lags'] + 1):
            features.append(f'temp_lag_{i}')

        # --- Train/test split for metrics only ---
        app.logger.info("Evaluating model on train/test split...")
        test_size = int(len(df) * (1 - config['train_percentage'] / 100))
        df_test = df.iloc[:test_size].copy()    # oldest 10% for evaluation
        df_train = df.iloc[test_size:].copy()   # newest 90% for training

        dynamic_config = None
        if config['use_dynamic_quantile']:
            dynamic_config = {
                'peak_start': config['peak_start'],
                'peak_end': config['peak_end'],
                'peak_quantile': config['peak_quantile'],
                'offpeak_quantile': config['offpeak_quantile']
            }

        eval_model = QuantileRegressionModel(quantile=config['quantile'], dynamic_config=dynamic_config)
        eval_model.train(df_train[features].values, df_train['consumption'].values, df_train['hour'].values)

        eval_result = predict_iterative(
            df_test[features].values, df_test['consumption'].values,
            eval_model, features, config['n_power_lags'], df_test['hour'].values
        )
        metrics = eval_model.evaluate(df_test['consumption'].values, eval_result['predictions'])
        coverage = eval_model.calculate_coverage(df_test['consumption'].values, eval_result['predictions'])

        app.logger.info(f"Model metrics — R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}, Coverage: {coverage:.1f}%")

        # --- Retrain on ALL historical data for best future predictions ---
        app.logger.info("Retraining on full dataset for future predictions...")
        final_model = QuantileRegressionModel(quantile=config['quantile'], dynamic_config=dynamic_config)
        final_model.train(df[features].values, df['consumption'].values, df['hour'].values)

        # --- Build future feature matrix ---
        app.logger.info("Building future prediction features...")
        forecast_data = app_state.get('weather_forecast') or []
        df_future = _build_future_features(config, df, features, forecast_data)

        # --- Predict on future bins ---
        app.logger.info(f"Predicting {len(df_future)} future bins...")
        X_future = df_future[features].values
        hours_future = df_future['hour'].values

        # Use iterative prediction so power lags update as we step forward
        dummy_y = np.zeros(len(X_future))
        future_result = predict_iterative(
            X_future, dummy_y, final_model, features,
            config['n_power_lags'], hours_future
        )
        y_future = future_result['predictions']

        # --- Build predictions list with real future timestamps ---
        predictions_list = []
        for i, row in df_future.iterrows():
            predictions_list.append({
                'timestamp': row['timestamp'].isoformat(),
                'predicted': round(float(max(0, y_future[i])), 3)
            })

        app.logger.info(f"Generated {len(predictions_list)} future predictions "
                        f"from {predictions_list[0]['timestamp']} to {predictions_list[-1]['timestamp']}")

        prediction_results = {
            'predictions': predictions_list,
            'metrics': {
                'r2': float(metrics['r2']),
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'coverage': float(coverage)
            },
            'model_info': {
                'type': 'Quantile Regression',
                'quantile': config['quantile'],
                'use_dynamic': config['use_dynamic_quantile'],
                'peak_hours': f"{config['peak_start']}-{config['peak_end']}" if config['use_dynamic_quantile'] else None,
                'peak_quantile': config['peak_quantile'] if config['use_dynamic_quantile'] else None,
                'offpeak_quantile': config['offpeak_quantile'] if config['use_dynamic_quantile'] else None,
                'trained_on_samples': len(df),
                'forecast_start': predictions_list[0]['timestamp'],
                'forecast_end': predictions_list[-1]['timestamp']
            },
            'data_info': {
                'train_samples': int(len(df_train)),
                'test_samples': int(len(df_test)),
                'features': features,
                'history_days': config['history_days']
            }
        }

        app_state['prediction_results'] = prediction_results
        app_state['last_update']['prediction'] = datetime.now().isoformat()

        # --- Publish to Home Assistant ---
        try:
            ha_client = HomeAssistantClient(
                base_url=os.getenv('HA_URL', 'http://supervisor/core'),
                token=os.getenv('SUPERVISOR_TOKEN')
            )
            ha_client.create_prediction_sensors(predictions_list, config['power_entity'])
            app.logger.info("Published future predictions to Home Assistant")
        except Exception as e:
            app.logger.error(f"Failed to publish to HA: {e}")

        return jsonify(prediction_results)

    except Exception as e:
        app.logger.error(f"Training error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get add-on status and data availability."""
    return jsonify({
        'status': 'running',
        'data_loaded': {
            'power': app_state['power_data'] is not None,
            'weather_history': app_state['weather_history'] is not None,
            'weather_forecast': app_state['weather_forecast'] is not None,
            'predictions': app_state['prediction_results'] is not None
        },
        'last_update': app_state['last_update']
    })


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get last prediction results."""
    if app_state['prediction_results'] is None:
        return jsonify({'error': 'No predictions available. Please train the model first.'}), 404
    return jsonify(app_state['prediction_results'])


def save_csvs(df_historical, predictions_list, config):
    """
    Save prediction and historical CSVs to /share directory.
    Saves both latest and timestamped versions.
    
    Args:
        df_historical: DataFrame with historical power data
        predictions_list: List of prediction dicts with timestamp and predicted values
        config: Configuration dict
    
    Returns:
        Dict with file paths
    """
    import os
    from datetime import datetime
    
    # Create output directory
    output_dir = '/share/ha_power_predictor'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    # === Save Predictions CSV ===
    predictions_latest = os.path.join(output_dir, 'predictions_latest.csv')
    predictions_timestamped = os.path.join(output_dir, f'predictions_{timestamp_str}.csv')
    
    with open(predictions_latest, 'w') as f:
        f.write('timestamp,predicted_kw\n')
        for pred in predictions_list:
            f.write(f"{pred['timestamp']},{pred['predicted']}\n")
    
    # Copy to timestamped version
    import shutil
    shutil.copy(predictions_latest, predictions_timestamped)
    
    # === Save Historical 48h CSV ===
    # Get last 48 hours of data
    df_48h = df_historical.tail(48).copy()
    
    historical_latest = os.path.join(output_dir, 'historical_48h_latest.csv')
    historical_timestamped = os.path.join(output_dir, f'historical_48h_{timestamp_str}.csv')
    
    with open(historical_latest, 'w') as f:
        f.write('timestamp,actual_kw\n')
        for _, row in df_48h.iterrows():
            f.write(f"{row['timestamp'].isoformat()},{row['consumption']}\n")
    
    shutil.copy(historical_latest, historical_timestamped)
    
    return {
        'predictions_latest': predictions_latest,
        'predictions_timestamped': predictions_timestamped,
        'historical_latest': historical_latest,
        'historical_timestamped': historical_timestamped
    }


@app.route('/api/run-prediction-pipeline', methods=['POST'])
def run_prediction_pipeline():
    """
    Full automated pipeline: fetch data → train → save CSVs → publish sensors.
    Designed to be called by HA automation every hour.
    """
    try:
        config = get_config_from_env()
        use_stats = config.get('use_hourly_statistics', False)
        
        ha_client = HomeAssistantClient(
            base_url=os.getenv('HA_URL', 'http://supervisor/core'),
            token=os.getenv('SUPERVISOR_TOKEN')
        )
        
        app.logger.info("=== PREDICTION PIPELINE START ===")
        
        # --- Step 1: Fetch Power Data ---
        app.logger.info("Step 1/4: Fetching power data...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=config['history_days'])
        
        if use_stats:
            power_data = ha_client.get_statistics(config['power_entity'], start_time, end_time, 'hour')
        else:
            power_data = ha_client.get_history(config['power_entity'], start_time, end_time)
        
        if not power_data:
            return jsonify({'success': False, 'error': 'No power data found'}), 400
        
        app_state['power_data'] = power_data
        
        # --- Step 2: Fetch Temperature Data ---
        app.logger.info("Step 2/4: Fetching temperature data...")
        if use_stats:
            temp_data = ha_client.get_statistics(config['temperature_entity'], start_time, end_time, 'hour')
        else:
            temp_data = ha_client.get_history(config['temperature_entity'], start_time, end_time)
        
        if not temp_data:
            return jsonify({'success': False, 'error': 'No temperature data found'}), 400
        
        app_state['weather_history'] = temp_data
        
        # --- Step 3: Fetch Weather Forecast ---
        app.logger.info("Step 3/4: Fetching weather forecast...")
        forecast_data = ha_client.get_weather_forecast(config['weather_forecast_entity'], 'hourly')
        app_state['weather_forecast'] = forecast_data or []
        
        # --- Step 4: Train Model & Generate Predictions ---
        app.logger.info("Step 4/4: Training model and generating predictions...")
        
        # Process historical data
        if use_stats:
            df = process_ha_statistics(power_data, temp_data, timezone=config['timezone'])
        else:
            df = process_ha_data(power_data, temp_data, bin_size_minutes=config['bin_size_minutes'], timezone=config['timezone'])
        
        if len(df) < 100:
            return jsonify({'success': False, 'error': f'Insufficient data: only {len(df)} records'}), 400
        
        df = add_lagged_features(df, n_power_lags=config['n_power_lags'], n_temp_lags=config['n_temp_lags'])
        
        features = get_default_features()
        for i in range(1, config['n_power_lags'] + 1):
            features.append(f'power_lag_{i}')
        for i in range(1, config['n_temp_lags'] + 1):
            features.append(f'temp_lag_{i}')
        
        # Train/test split for metrics
        test_size = int(len(df) * (1 - config['train_percentage'] / 100))
        df_test = df.iloc[:test_size].copy()
        df_train = df.iloc[test_size:].copy()
        
        dynamic_config = None
        if config['use_dynamic_quantile']:
            dynamic_config = {
                'peak_start': config['peak_start'],
                'peak_end': config['peak_end'],
                'peak_quantile': config['peak_quantile'],
                'offpeak_quantile': config['offpeak_quantile']
            }
        
        eval_model = QuantileRegressionModel(quantile=config['quantile'], dynamic_config=dynamic_config)
        eval_model.train(df_train[features].values, df_train['consumption'].values, df_train['hour'].values)
        
        eval_result = predict_iterative(
            df_test[features].values, df_test['consumption'].values,
            eval_model, features, config['n_power_lags'], df_test['hour'].values
        )
        metrics = eval_model.evaluate(df_test['consumption'].values, eval_result['predictions'])
        coverage = eval_model.calculate_coverage(df_test['consumption'].values, eval_result['predictions'])
        
        # Retrain on all data
        final_model = QuantileRegressionModel(quantile=config['quantile'], dynamic_config=dynamic_config)
        final_model.train(df[features].values, df['consumption'].values, df['hour'].values)
        
        # Build future predictions
        df_future = _build_future_features(config, df, features, forecast_data)
        X_future = df_future[features].values
        hours_future = df_future['hour'].values
        dummy_y = np.zeros(len(X_future))
        future_result = predict_iterative(X_future, dummy_y, final_model, features, config['n_power_lags'], hours_future)
        y_future = future_result['predictions']
        
        predictions_list = []
        for i, row in df_future.iterrows():
            predictions_list.append({
                'timestamp': row['timestamp'].isoformat(),
                'predicted': round(float(max(0, y_future[i])), 3)
            })
        
        # --- Save CSVs ---
        app.logger.info("Saving CSVs...")
        csv_files = save_csvs(df, predictions_list, config)
        
        # --- Publish to Home Assistant ---
        app.logger.info("Publishing sensors to Home Assistant...")
        ha_client.create_prediction_sensors(predictions_list, config['power_entity'])
        
        # --- Store results ---
        app_state['prediction_results'] = {
            'predictions': predictions_list,
            'metrics': {
                'r2': float(metrics['r2']),
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'coverage': float(coverage)
            }
        }
        
        app.logger.info("=== PIPELINE COMPLETE ===")
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'r2': float(metrics['r2']),
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'coverage': float(coverage)
            },
            'files': csv_files,
            'record_counts': {
                'predictions': len(predictions_list),
                'historical_48h': min(48, len(df))
            }
        })
        
    except Exception as e:
        app.logger.error(f"Pipeline error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=False, use_reloader=False)
