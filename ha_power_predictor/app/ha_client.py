"""
Home Assistant API client for reading history and writing predictions.
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Client for interacting with Home Assistant API."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

    def get_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime = None
    ) -> List[Dict[str, Any]]:
        if end_time is None:
            end_time = datetime.now()

        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')

        url = f'{self.base_url}/api/history/period/{start_str}'
        params = {
            'filter_entity_id': entity_id,
            'end_time': end_str,
            'minimal_response': 'true',
            'significant_changes_only': 'true'
        }

        logger.info(f"Fetching history for {entity_id} from {start_str} to {end_str}")

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=300)
            response.raise_for_status()

            data = response.json()

            if not data or not isinstance(data, list) or len(data) == 0:
                logger.warning(f"No history data returned for {entity_id}")
                return []

            entity_history = data[0] if isinstance(data[0], list) else data
            logger.info(f"Retrieved {len(entity_history)} history records")
            return entity_history

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch history: {e}")
            raise

    def get_statistics(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime = None,
        period: str = 'hour'
    ) -> List[Dict[str, Any]]:
        """
        Fetch hourly statistics from Home Assistant.
        This returns pre-aggregated hourly data (mean, min, max) which is much more efficient
        than fetching all raw state changes.
        
        Args:
            entity_id: Sensor entity ID
            start_time: Start of the time window
            end_time: End of the time window (defaults to now)
            period: Statistics period ('hour', 'day', 'week', 'month')
        
        Returns:
            List of statistics records with 'start', 'mean', 'min', 'max' fields
        """
        if end_time is None:
            end_time = datetime.now()

        # Format: /api/history/period/{start_time} with entity filter
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
        
        url = f'{self.base_url}/api/history/period/{start_str}'
        params = {
            'filter_entity_id': entity_id,
            'end_time': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'minimal_response': 'true',
            'no_attributes': 'true'
        }

        logger.info(f"Fetching statistics-style history for {entity_id} from {start_str}")
        logger.info(f"Note: Using history API with minimal_response for efficiency")

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=300)
            response.raise_for_status()

            data = response.json()

            if not data or not isinstance(data, list) or len(data) == 0:
                logger.warning(f"No history data returned for {entity_id}")
                return []

            entity_history = data[0] if isinstance(data[0], list) else data
            
            # Convert to hourly bins on the fly
            import pandas as pd
            records = []
            for record in entity_history:
                try:
                    records.append({
                        'last_changed': record.get('last_changed') or record.get('last_updated'),
                        'state': record.get('state')
                    })
                except:
                    continue
            
            if not records:
                return []
            
            # Create DataFrame and bin to hourly
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['last_changed'])
            df['state'] = pd.to_numeric(df['state'], errors='coerce')
            df = df.dropna(subset=['state'])
            
            # Bin to hourly
            df['hour_bin'] = df['timestamp'].dt.floor('1H')
            hourly = df.groupby('hour_bin')['state'].agg(['mean', 'min', 'max']).reset_index()
            
            # Convert to statistics format
            stats = []
            for _, row in hourly.iterrows():
                stats.append({
                    'start': row['hour_bin'].isoformat(),
                    'mean': float(row['mean']),
                    'min': float(row['min']),
                    'max': float(row['max'])
                })
            
            logger.info(f"Aggregated {len(entity_history)} records into {len(stats)} hourly bins")
            return stats

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch statistics: {e}")
            raise

    def get_weather_forecast(
        self,
        entity_id: str,
        forecast_type: str = 'hourly'
    ) -> List[Dict[str, Any]]:
        url = f'{self.base_url}/api/services/weather/get_forecasts?return_response'

        payload = {
            'entity_id': entity_id,
            'type': forecast_type
        }

        logger.info(f"Fetching {forecast_type} weather forecast for {entity_id}")

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)

            logger.info(f"Response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Response text: {response.text}")

            response.raise_for_status()

            data = response.json()
            logger.info(f"Response data keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")

            # Extract forecast from response.
            # HA 2024.2+ wraps the result in a 'service_response' key:
            # {'service_response': {entity_id: {'forecast': [...]}}}
            forecasts = []

            if isinstance(data, dict):
                # HA 2024.2+ format
                if 'service_response' in data:
                    service_data = data['service_response']
                    if entity_id in service_data:
                        forecasts = service_data[entity_id].get('forecast', [])
                    else:
                        # Fall back to first entity in service_response
                        for value in service_data.values():
                            if isinstance(value, dict) and 'forecast' in value:
                                forecasts = value['forecast']
                                break
                # Legacy direct entity key
                elif entity_id in data:
                    forecasts = data[entity_id].get('forecast', [])
                # Legacy flat forecast key
                elif 'forecast' in data:
                    forecasts = data['forecast']

            if forecasts:
                logger.info(f"Retrieved {len(forecasts)} forecast records")
            else:
                logger.warning(f"No forecast data found in response: {data}")

            return forecasts

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch weather forecast: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise

    def set_state(
        self,
        entity_id: str,
        state: Any,
        attributes: Dict[str, Any] = None
    ) -> bool:
        url = f'{self.base_url}/api/states/{entity_id}'

        payload = {
            'state': state,
            'attributes': attributes or {}
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Set state for {entity_id}: {state}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to set state: {e}")
            return False

    def create_prediction_sensors(
        self,
        predictions: List[Dict[str, Any]],
        source_entity: str
    ) -> bool:
        if not predictions:
            return False

        windows = {
            '1h': 1,
            '6h': 6,
            '12h': 12,
            '24h': 24,
            '48h': 48
        }

        success = True

        for window_name, hours in windows.items():
            if len(predictions) < hours:
                continue

            window_predictions = predictions[:hours]
            avg_prediction = sum(p['predicted'] for p in window_predictions) / len(window_predictions)
            max_prediction = max(p['predicted'] for p in window_predictions)

            entity_id = f'sensor.power_prediction_next_{window_name}'

            attributes = {
                'unit_of_measurement': 'kW',
                'friendly_name': f'Power Prediction Next {window_name.upper()}',
                'icon': 'mdi:flash',
                'device_class': 'power',
                'source_entity': source_entity,
                'window': window_name,
                'average': round(avg_prediction, 2),
                'maximum': round(max_prediction, 2),
                'predictions': [
                    {'time': p['timestamp'], 'value': round(p['predicted'], 2)}
                    for p in window_predictions
                ]
            }

            if not self.set_state(entity_id, round(avg_prediction, 2), attributes):
                success = False

        # Full prediction dataset sensor
        self.set_state(
            'sensor.power_prediction_full',
            len(predictions),
            {
                'friendly_name': 'Power Prediction Full Dataset',
                'icon': 'mdi:chart-line',
                'source_entity': source_entity,
                'predictions': [
                    {
                        'time': p['timestamp'],
                        'predicted': round(p['predicted'], 2),
                        'actual': round(p.get('actual', 0), 2) if p.get('actual') else None
                    }
                    for p in predictions
                ],
                'count': len(predictions)
            }
        )

        return success
