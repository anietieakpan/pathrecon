# scripts/test_databases.py

import sys
import os
from datetime import datetime
import pytz
import logging
from config import Config
from app.database.factory import DatabaseFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_influxdb():
    """Test InfluxDB connection and data insertion"""
    logger.info("Testing InfluxDB connection...")
    
    try:
        db = DatabaseFactory.get_database('timeseries')
        db.connect()
        
        # Test data insertion
        test_data = {
            'text': 'TEST123',
            'confidence': 0.95,
            'timestamp_utc': datetime.now(pytz.UTC),
            'timestamp_local': datetime.now(pytz.UTC).astimezone(pytz.timezone('Africa/Johannesburg'))
        }
        
        db.insert_detection(test_data)
        logger.info("Successfully inserted test data into InfluxDB")
        
        # Test data retrieval
        end_time = datetime.now(pytz.UTC)
        start_time = end_time.replace(minute=end_time.minute - 5)  # Last 5 minutes
        results = db.get_detections(start_time, end_time)
        
        logger.info(f"Retrieved {len(results)} records from InfluxDB")
        
        db.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"InfluxDB test failed: {str(e)}")
        return False

def test_postgres():
    """Test PostgreSQL connection and data insertion"""
    logger.info("Testing PostgreSQL connection...")
    
    try:
        conn = DatabaseFactory.get_database('postgres')
        cursor = conn.cursor()
        
        # Test data insertion
        test_data = {
            'text': 'TEST123',
            'confidence': 0.95,
            'timestamp_utc': datetime.now(pytz.UTC),
            'timestamp_local': datetime.now(pytz.UTC).astimezone(pytz.timezone('Africa/Johannesburg'))
        }
        
        cursor.execute("""
            INSERT INTO analysis.vehicle_detections 
            (plate_text, confidence, timestamp_utc, timestamp_local)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """, (
            test_data['text'],
            test_data['confidence'],
            test_data['timestamp_utc'],
            test_data['timestamp_local']
        ))
        
        inserted_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Successfully inserted test data into PostgreSQL with ID: {inserted_id}")
        
        # Test data retrieval
        cursor.execute("""
            SELECT plate_text, confidence, timestamp_utc, timestamp_local
            FROM analysis.vehicle_detections
            WHERE id = %s
        """, (inserted_id,))
        
        result = cursor.fetchone()
        logger.info(f"Retrieved test record: {result}")
        
        # Clean up test data
        cursor.execute("DELETE FROM analysis.vehicle_detections WHERE id = %s", (inserted_id,))
        conn.commit()
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"PostgreSQL test failed: {str(e)}")
        if 'conn' in locals() and conn is not None:
            conn.rollback()
            conn.close()
        return False

def run_tests():
    """Run all database tests"""
    logger.info("Starting database tests...")
    
    results = {
        'InfluxDB': test_influxdb(),
        'PostgreSQL': test_postgres()
    }
    
    # Print results
    logger.info("\nTest Results:")
    for db_name, success in results.items():
        logger.info(f"{db_name}: {'PASSED' if success else 'FAILED'}")
    
    # Return overall success
    return all(results.values())

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)