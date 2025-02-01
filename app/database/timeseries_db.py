# app/database/timeseries_db.py

import logging
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from .base import DatabaseInterface
from typing import Dict, Any  # Added this import

logger = logging.getLogger(__name__)

class TimeSeriesDB(DatabaseInterface):
    def __init__(self, url, token, org, bucket):
        super().__init__()
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = None
        self.write_api = None
        self._connected = False  # Add this line
        
        
    def is_connected(self) -> bool:
        """Check if connection is active"""
        if not self.client:
            return False
        try:
            # Try to ping the server
            self.client.ping()
            return True
        except Exception:
            return False
    

    def connect(self):
        """Connect to InfluxDB"""
        try:
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            logger.info("Successfully connected to InfluxDB")
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {str(e)}")
            raise

    def disconnect(self):
        """Disconnect from InfluxDB"""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.write_api = None
            logger.info("Disconnected from InfluxDB")
        except Exception as e:
            logger.error(f"Error disconnecting from InfluxDB: {str(e)}")

    def insert_detection(self, detection_data):
        """Insert detection data into InfluxDB"""
        try:
            if not self.client or not self.write_api:
                self.connect()

            # Create point
            point = Point("license_plate_detection")
            
            # Add fields
            point.field("plate_text", str(detection_data['text']))
            point.field("confidence", float(detection_data['confidence']))
            
            # Handle timestamp
            if isinstance(detection_data.get('timestamp_utc'), datetime):
                point.time(detection_data['timestamp_utc'])
            
            # Add vehicle details if available
            vehicle_details = detection_data.get('vehicle_details', {})
            if vehicle_details:
                for key in ['make', 'model', 'color', 'type']:
                    if vehicle_details.get(key):
                        point.field(f"vehicle_{key}", str(vehicle_details[key]))
                if vehicle_details.get('year'):
                    point.field("vehicle_year", int(vehicle_details['year']))

            # Write point
            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=point
            )
            
            logger.debug(f"Successfully inserted detection: {detection_data['text']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting data into InfluxDB: {str(e)}")
            return False

            
            
    def _insert_detection_impl(self, detection_data: Dict[str, Any]) -> None:
        """Insert detection data into InfluxDB"""
        try:
            # Create point
            point = Point("license_plate_detection")
            
            # Add fields
            point.field("plate_text", str(detection_data['text']))
            point.field("confidence", float(detection_data['confidence']))
            
            # Handle timestamp
            if isinstance(detection_data.get('timestamp_utc'), datetime):
                point.time(detection_data['timestamp_utc'])
            
            # Add vehicle details if available
            vehicle_details = detection_data.get('vehicle_details', {})
            if vehicle_details:
                for key in ['make', 'model', 'color', 'type']:
                    if vehicle_details.get(key):
                        point.field(f"vehicle_{key}", str(vehicle_details[key]))
                if vehicle_details.get('year'):
                    point.field("vehicle_year", int(vehicle_details['year']))

            # Write point
            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=point
            )
            
            logger.debug(f"Successfully inserted detection: {detection_data['text']}")
            
        except Exception as e:
            logger.error(f"Error inserting data into InfluxDB: {str(e)}")
            raise
            

    def update_detection(self, detection_id, update_data):
        """
        Update is not supported in InfluxDB as it's an append-only time series database.
        Instead, we insert a new point with updated values.
        """
        try:
            if not self.client or not self.write_api:
                self.connect()

            # Create new point with updated data
            point = Point("license_plate_detection")
            point.field("detection_id", str(detection_id))
            
            for key, value in update_data.items():
                if isinstance(value, (int, float)):
                    point.field(key, value)
                else:
                    point.field(key, str(value))

            point.time(datetime.utcnow())
            
            # Write point
            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=point
            )
            
            logger.debug(f"Successfully added update point for detection: {detection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating detection in InfluxDB: {str(e)}")
            return False

    def delete_detection(self, detection_id):
        """
        Delete is not directly supported in InfluxDB.
        We can mark records as deleted by writing a delete marker.
        """
        try:
            if not self.client or not self.write_api:
                self.connect()

            # Create delete marker point
            point = Point("license_plate_detection")
            point.field("detection_id", str(detection_id))
            point.field("deleted", True)
            point.time(datetime.utcnow())
            
            # Write delete marker
            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=point
            )
            
            logger.debug(f"Successfully marked detection as deleted: {detection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error marking detection as deleted in InfluxDB: {str(e)}")
            return False

    def get_detections(self, start_time, end_time):
        """Get detections from InfluxDB within time range"""
        try:
            if not self.client:
                self.connect()

            query = f'''
                from(bucket:"{self.bucket}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r["_measurement"] == "license_plate_detection")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> filter(fn: (r) => r["deleted"] != true)
            '''

            results = []
            query_api = self.client.query_api()
            tables = query_api.query(query, org=self.org)
            
            for table in tables:
                for record in table.records:
                    results.append({
                        'text': record.values.get('plate_text', ''),
                        'confidence': float(record.values.get('confidence', 0.0)),
                        'timestamp_utc': record.get_time(),
                        'vehicle_details': {
                            'make': record.values.get('vehicle_make'),
                            'model': record.values.get('vehicle_model'),
                            'color': record.values.get('vehicle_color'),
                            'type': record.values.get('vehicle_type'),
                            'year': record.values.get('vehicle_year')
                        }
                    })

            return results
            
        except Exception as e:
            logger.error(f"Error querying InfluxDB: {str(e)}")
            return []