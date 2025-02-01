# app/database/postgres_db.py

import psycopg2
from psycopg2.extras import DictCursor, Json
from .base import DatabaseInterface
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PostgresDB(DatabaseInterface):
    def __init__(self, dbname, user, password, host, port):
        super().__init__()  # Add this line
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self.cursor = None
        self._connected = False  # Add this line

        
    def is_connected(self) -> bool:
        """Check if connection is active"""
        if not self.conn or self.conn.closed:
            return False
        try:
            # Try a simple query to test connection
            self.cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

    def connect(self):
        """Establish connection to PostgreSQL"""
        try:
            if not self.conn or self.conn.closed:
                self.conn = psycopg2.connect(
                    dbname=self.dbname,
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port
                )
                self.cursor = self.conn.cursor(cursor_factory=DictCursor)
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise

    def insert_detection(self, detection_data: Dict[str, Any]) -> int:
        """Insert detection with vehicle details into PostgreSQL"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()

            query = """
                INSERT INTO analysis.vehicle_detections 
                (plate_text, confidence, timestamp_utc, timestamp_local,
                 vehicle_make, vehicle_model, vehicle_color, vehicle_year,
                 vehicle_type, vehicle_image_path, vehicle_confidence_scores,
                 camera_id, location, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            
            # Extract vehicle details if they exist
            vehicle_details = detection_data.get('vehicle_details', {})
            
            values = (
                detection_data['text'],
                detection_data['confidence'],
                detection_data['timestamp_utc'],
                detection_data['timestamp_local'],
                vehicle_details.get('make'),
                vehicle_details.get('model'),
                vehicle_details.get('color'),
                vehicle_details.get('year'),
                vehicle_details.get('type'),
                vehicle_details.get('image_path'),
                Json(vehicle_details.get('confidence_scores')) if vehicle_details.get('confidence_scores') else None,
                detection_data.get('camera_id'),
                Json(detection_data.get('location')) if detection_data.get('location') else None,
                Json(detection_data.get('metadata')) if detection_data.get('metadata') else None
            )
            
            self.cursor.execute(query, values)
            inserted_id = self.cursor.fetchone()[0]
            self.conn.commit()
            
            logger.debug(f"Inserted detection with ID {inserted_id}")
            return inserted_id
            
        except Exception as e:
            if self.conn and not self.conn.closed:
                self.conn.rollback()
            logger.error(f"Error inserting detection: {str(e)}")
            raise
        
        
    def _insert_detection_impl(self, detection_data: Dict[str, Any]) -> None:
        """Insert detection with vehicle details into PostgreSQL"""
        try:
            query = """
                INSERT INTO analysis.vehicle_detections 
                (plate_text, confidence, timestamp_utc, timestamp_local,
                vehicle_make, vehicle_model, vehicle_color, vehicle_year,
                vehicle_type, vehicle_image_path, vehicle_confidence_scores,
                camera_id, location, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            
            # Extract vehicle details if they exist
            vehicle_details = detection_data.get('vehicle_details', {})
            
            values = (
                detection_data['text'],
                detection_data['confidence'],
                detection_data['timestamp_utc'],
                detection_data['timestamp_local'],
                vehicle_details.get('make'),
                vehicle_details.get('model'),
                vehicle_details.get('color'),
                vehicle_details.get('year'),
                vehicle_details.get('type'),
                vehicle_details.get('image_path'),
                Json(vehicle_details.get('confidence_scores')) if vehicle_details.get('confidence_scores') else None,
                detection_data.get('camera_id'),
                Json(detection_data.get('location')) if detection_data.get('location') else None,
                Json(detection_data.get('metadata')) if detection_data.get('metadata') else None
            )
            
            self.cursor.execute(query, values)
            inserted_id = self.cursor.fetchone()[0]
            self.conn.commit()
            
            logger.debug(f"Inserted detection with ID {inserted_id}")
            return inserted_id
                
        except Exception as e:
            if self.conn and not self.conn.closed:
                self.conn.rollback()
            logger.error(f"Error inserting detection: {str(e)}")
            raise
        
        
        

    def get_detections(self, start_time: datetime, end_time: datetime, 
                      include_vehicle_details: bool = True) -> List[Dict[str, Any]]:
        """Get detections with optional vehicle details"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()

            query = """
                SELECT id, plate_text, confidence, timestamp_utc, timestamp_local,
                       vehicle_make, vehicle_model, vehicle_color, vehicle_year,
                       vehicle_type, vehicle_image_path, vehicle_confidence_scores,
                       camera_id, location, metadata
                FROM analysis.vehicle_detections
                WHERE timestamp_utc BETWEEN %s AND %s
                ORDER BY timestamp_utc DESC;
            """

            self.cursor.execute(query, (start_time, end_time))
            results = self.cursor.fetchall()

            detections = []
            for row in results:
                detection = {
                    'id': row['id'],
                    'text': row['plate_text'],
                    'confidence': row['confidence'],
                    'timestamp_utc': row['timestamp_utc'],
                    'timestamp_local': row['timestamp_local'],
                    'camera_id': row['camera_id'],
                    'location': row['location'],
                    'metadata': row['metadata']
                }
                
                if include_vehicle_details:
                    detection['vehicle_details'] = {
                        'make': row['vehicle_make'],
                        'model': row['vehicle_model'],
                        'color': row['vehicle_color'],
                        'year': row['vehicle_year'],
                        'type': row['vehicle_type'],
                        'image_path': row['vehicle_image_path'],
                        'confidence_scores': row['vehicle_confidence_scores']
                    }
                
                detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error getting detections: {str(e)}")
            raise

    def get_vehicle_statistics(self) -> Dict[str, Any]:
        """Get vehicle statistics from materialized view"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()

            self.cursor.execute("""
                SELECT * FROM mv_vehicle_statistics 
                ORDER BY detection_count DESC;
            """)
            
            results = self.cursor.fetchall()
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error getting vehicle statistics: {str(e)}")
            raise

    def refresh_statistics(self):
        """Manually refresh vehicle statistics"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()

            self.cursor.execute("SELECT refresh_vehicle_statistics();")
            self.conn.commit()
            logger.info("Successfully refreshed vehicle statistics")

        except Exception as e:
            logger.error(f"Error refreshing statistics: {str(e)}")
            raise

    def update_detection(self, detection_id: int, update_data: Dict[str, Any]) -> bool:
        """Update an existing detection with vehicle details"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()

            update_fields = []
            update_values = []
            
            # Map of field names to their database column names
            field_mapping = {
                'text': 'plate_text',
                'confidence': 'confidence',
                'make': 'vehicle_make',
                'model': 'vehicle_model',
                'color': 'vehicle_color',
                'year': 'vehicle_year',
                'type': 'vehicle_type',
                'image_path': 'vehicle_image_path',
                'confidence_scores': 'vehicle_confidence_scores'
            }
            
            for field, value in update_data.items():
                if field in field_mapping:
                    column = field_mapping[field]
                    update_fields.append(f"{column} = %s")
                    update_values.append(
                        Json(value) if field == 'confidence_scores' else value
                    )
            
            if not update_fields:
                logger.warning("No fields to update")
                return False

            update_values.append(detection_id)
            
            query = f"""
                UPDATE analysis.vehicle_detections
                SET {', '.join(update_fields)}
                WHERE id = %s
                RETURNING id;
            """
            
            self.cursor.execute(query, update_values)
            updated = self.cursor.fetchone()
            self.conn.commit()
            
            success = updated is not None
            if success:
                logger.debug(f"Updated detection ID {detection_id}")
            else:
                logger.warning(f"No detection found with ID {detection_id}")
            
            return success
            
        except Exception as e:
            if self.conn and not self.conn.closed:
                self.conn.rollback()
            logger.error(f"Error updating detection: {str(e)}")
            raise
    
    
    def delete_detection(self, detection_id):
        """Delete a detection"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()

            self.cursor.execute("""
                DELETE FROM analysis.vehicle_detections
                WHERE id = %s
                RETURNING id;
            """, (detection_id,))
            
            deleted = self.cursor.fetchone()
            self.conn.commit()
            
            success = deleted is not None
            if success:
                logger.debug(f"Deleted detection ID {detection_id}")
            else:
                logger.warning(f"No detection found with ID {detection_id}")
            
            return success
            
        except Exception as e:
            if self.conn and not self.conn.closed:
                self.conn.rollback()
            logger.error(f"Error deleting detection from PostgreSQL: {str(e)}")
            raise
        
    def disconnect(self):
        """Close PostgreSQL connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
                self.conn = None
                self.cursor = None
            logger.info("Disconnected from PostgreSQL database")
        except Exception as e:
            logger.error(f"Error disconnecting from PostgreSQL: {str(e)}")

    def get_vehicle_makes_and_models(self):
        """Get hierarchical list of vehicle makes and models"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()

            query = """
            WITH make_stats AS (
                SELECT 
                    vehicle_make,
                    COUNT(DISTINCT vehicle_model) as model_count,
                    COUNT(*) as total_detections
                FROM analysis.vehicle_detections
                WHERE vehicle_make IS NOT NULL
                GROUP BY vehicle_make
            ),
            model_stats AS (
                SELECT 
                    vehicle_make,
                    vehicle_model,
                    COUNT(*) as detections,
                    array_agg(DISTINCT vehicle_year) as years,
                    array_agg(DISTINCT vehicle_type) as types
                FROM analysis.vehicle_detections
                WHERE vehicle_make IS NOT NULL 
                AND vehicle_model IS NOT NULL
                GROUP BY vehicle_make, vehicle_model
            )
            SELECT 
                jsonb_build_object(
                    'makes', jsonb_object_agg(
                        ms.vehicle_make,
                        jsonb_build_object(
                            'model_count', ms.model_count,
                            'total_detections', ms.total_detections,
                            'models', (
                                SELECT jsonb_object_agg(
                                    mods.vehicle_model,
                                    jsonb_build_object(
                                        'detections', mods.detections,
                                        'years', mods.years,
                                        'types', mods.types
                                    )
                                )
                                FROM model_stats mods
                                WHERE mods.vehicle_make = ms.vehicle_make
                            )
                        )
                    )
                ) as makes_and_models
            FROM make_stats ms;
            """

            self.cursor.execute(query)
            result = self.cursor.fetchone()
        
            return result[0] if result else {'makes': {}}

        except Exception as e:
            logger.error(f"Error getting vehicle makes and models: {str(e)}")
            return {'makes': {}}
        
    def get_vehicle_colors(self):
        """Get list of detected vehicle colors with counts"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()

            query = """
            SELECT 
                vehicle_color,
                COUNT(*) as count,
                COUNT(DISTINCT plate_text) as unique_vehicles,
                MIN(timestamp_utc) as first_seen,
                MAX(timestamp_utc) as last_seen,
                array_agg(DISTINCT vehicle_make) as makes
            FROM analysis.vehicle_detections
            WHERE vehicle_color IS NOT NULL
            GROUP BY vehicle_color
            ORDER BY count DESC;
            """

            self.cursor.execute(query)
            results = self.cursor.fetchall()

            colors = []
            for row in results:
                colors.append({
                    'color': row[0],
                    'count': row[1],
                    'unique_vehicles': row[2],
                    'first_seen': row[3].isoformat() if row[3] else None,
                    'last_seen': row[4].isoformat() if row[4] else None,
                    'makes': row[5]
                })

            return colors

        except Exception as e:
            logger.error(f"Error getting vehicle colors: {str(e)}")
            return []
        
    def search_vehicles(self, conditions: list, params: dict):
        """Search vehicles based on specified criteria"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()

            # Build the WHERE clause
            where_clause = " AND ".join(conditions) if conditions else "TRUE"
        
            query = f"""
            SELECT 
                plate_text,
                vehicle_make,
                vehicle_model,
                vehicle_color,
                vehicle_type,
                vehicle_year,
                vehicle_image_path,
                vehicle_confidence_scores,
                COUNT(*) as detection_count,
                MIN(timestamp_utc) as first_seen,
                MAX(timestamp_utc) as last_seen,
                array_agg(DISTINCT camera_id) as cameras
            FROM analysis.vehicle_detections
            WHERE {where_clause}
            AND timestamp_utc BETWEEN :start_time AND :end_time
            GROUP BY 
                plate_text, vehicle_make, vehicle_model, 
                vehicle_color, vehicle_type, vehicle_year,
                vehicle_image_path, vehicle_confidence_scores
            ORDER BY detection_count DESC;
            """

            # Convert params dict to format expected by psycopg2
            query = query.replace(':start_time', '%s')
            query = query.replace(':end_time', '%s')
            for key in params:
                if key not in ('start_time', 'end_time'):
                    query = query.replace(f':{key}', '%s')

            # Create ordered list of parameters
            param_values = []
            param_values.extend([params['start_time'], params['end_time']])
            param_values.extend([params[key] for key in params if key not in ('start_time', 'end_time')])

            self.cursor.execute(query, param_values)
            results = self.cursor.fetchall()

            vehicles = []
            for row in results:
                vehicles.append({
                    'plate_text': row[0],
                    'vehicle_details': {
                        'make': row[1],
                        'model': row[2],
                        'color': row[3],
                        'type': row[4],
                        'year': row[5],
                        'image_path': row[6],
                        'confidence_scores': row[7]
                    },
                    'statistics': {
                        'detection_count': row[8],
                        'first_seen': row[9].isoformat() if row[9] else None,
                        'last_seen': row[10].isoformat() if row[10] else None,
                        'cameras': row[11]
                    }
                })

            return vehicles

        except Exception as e:
            logger.error(f"Error searching vehicles: {str(e)}")
            return []
        
        
    
    
    

            
    
        

