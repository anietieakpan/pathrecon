# tests/system_test.py

import sys
import os
from datetime import datetime, timedelta
import cv2
import psycopg2
from influxdb_client import InfluxDBClient
from config import Config
from app.detection.detector import LicensePlateDetector
from app.database.factory import DatabaseFactory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTest:
    def __init__(self):
        self.detector = None
        self.timeseries_db = None
        self.postgres_conn = None
        self.test_image_path = "data/examples/oneline_images/example1.jpeg"

    def setup(self):
        """Initialize all system components"""
        try:
            # Initialize detector
            self.detector = LicensePlateDetector(DatabaseFactory)
            self.detector.initialize_databases()
            logger.info("Detector initialized successfully")

            # Test PostgreSQL Connection
            self.postgres_conn = psycopg2.connect(
                dbname=Config.POSTGRES_DB,
                user=Config.POSTGRES_USER,
                password=Config.POSTGRES_PASSWORD,
                host=Config.POSTGRES_HOST,
                port=Config.POSTGRES_PORT
            )
            logger.info("PostgreSQL connection successful")

            # Test InfluxDB Connection
            self.influx_client = InfluxDBClient(
                url=Config.INFLUXDB_URL,
                token=Config.INFLUXDB_TOKEN,
                org=Config.INFLUXDB_ORG
            )
            health = self.influx_client.health()
            if health.status == "pass":
                logger.info("InfluxDB connection successful")
            else:
                logger.error("InfluxDB health check failed")

            return True

        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            return False

    def test_detection(self):
        """Test license plate detection"""
        try:
            if not os.path.exists(self.test_image_path):
                logger.error(f"Test image not found: {self.test_image_path}")
                return False

            image = cv2.imread(self.test_image_path)
            encoded_image, detections = self.detector.process_image(image)

            if not detections:
                logger.error("No license plates detected in test image")
                return False

            logger.info(f"Successfully detected {len(detections)} license plates")
            for det in detections:
                logger.info(f"Plate: {det['text']}, Confidence: {det['confidence']:.2f}")

            return True

        except Exception as e:
            logger.error(f"Detection test failed: {str(e)}")
            return False

    def test_data_flow(self):
        """Test data flow through the system"""
        try:
            # 1. Process test image
            image = cv2.imread(self.test_image_path)
            _, detections = self.detector.process_image(image)

            if not detections:
                logger.error("No detections to test data flow")
                return False

            # 2. Query InfluxDB to verify data
            query_api = self.influx_client.query_api()
            query = f'''
                from(bucket:"{Config.INFLUXDB_BUCKET}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "license_plate_detection")
                |> tail(n: 1)
            '''
            result = query_api.query(query)
            if not result:
                logger.error("No data found in InfluxDB")
                return False

            # 3. Query PostgreSQL to verify replication
            cursor = self.postgres_conn.cursor()
            cursor.execute("""
                SELECT plate_text, confidence, timestamp_utc 
                FROM analysis.vehicle_detections 
                ORDER BY timestamp_utc DESC 
                LIMIT 1
            """)
            postgres_result = cursor.fetchone()
            
            if not postgres_result:
                logger.error("No data found in PostgreSQL")
                return False

            logger.info("Data flow test successful")
            logger.info(f"Latest PostgreSQL record: {postgres_result}")
            return True

        except Exception as e:
            logger.error(f"Data flow test failed: {str(e)}")
            return False

    def verify_data_consistency(self):
        """Verify data consistency between InfluxDB and PostgreSQL"""
        try:
            # Get count from InfluxDB for last hour
            query_api = self.influx_client.query_api()
            query = f'''
                from(bucket:"{Config.INFLUXDB_BUCKET}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "license_plate_detection")
                |> count()
            '''
            influx_result = query_api.query(query)
            influx_count = sum(table.records[0].get_value() for table in influx_result)

            # Get count from PostgreSQL for last hour
            cursor = self.postgres_conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) 
                FROM analysis.vehicle_detections 
                WHERE timestamp_utc >= NOW() - INTERVAL '1 hour'
            """)
            postgres_count = cursor.fetchone()[0]

            logger.info(f"InfluxDB count: {influx_count}")
            logger.info(f"PostgreSQL count: {postgres_count}")

            return abs(influx_count - postgres_count) <= 1  # Allow for slight timing differences

        except Exception as e:
            logger.error(f"Data consistency check failed: {str(e)}")
            return False

    def cleanup(self):
        """Cleanup connections and resources"""
        try:
            if self.postgres_conn:
                self.postgres_conn.close()
            if self.influx_client:
                self.influx_client.close()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

def run_system_test():
    """Run all system tests"""
    test = SystemTest()
    
    try:
        # Setup
        logger.info("Starting system test...")
        if not test.setup():
            logger.error("Setup failed")
            return False

        # Run tests
        tests = [
            ("Detection Test", test.test_detection),
            ("Data Flow Test", test.test_data_flow),
            ("Data Consistency Test", test.verify_data_consistency)
        ]

        results = []
        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name}...")
            result = test_func()
            results.append((test_name, result))
            logger.info(f"{test_name}: {'PASSED' if result else 'FAILED'}")

        # Print summary
        logger.info("\nTest Summary:")
        for test_name, result in results:
            logger.info(f"{test_name}: {'PASSED' if result else 'FAILED'}")

        return all(result for _, result in results)

    except Exception as e:
        logger.error(f"System test failed: {str(e)}")
        return False
    finally:
        test.cleanup()

if __name__ == "__main__":
    success = run_system_test()
    sys.exit(0 if success else 1)