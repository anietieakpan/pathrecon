# scripts/verify_db.py

import psycopg2
from psycopg2.extras import DictCursor
from config import Config
import logging

logger = logging.getLogger(__name__)

def verify_database_structure():
    """Verify the database structure after migration"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            dbname=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD,
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT
        )
        cursor = conn.cursor(cursor_factory=DictCursor)

        # Check for vehicle-related columns
        logger.info("Checking table columns...")
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'analysis'
            AND table_name = 'vehicle_detections'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        print("\nTable Columns:")
        print("-------------")
        for col in columns:
            print(f"Column: {col['column_name']}")
            print(f"Type: {col['data_type']}")
            print(f"Nullable: {col['is_nullable']}")
            print("-------------")

        # Check for indexes
        logger.info("Checking indexes...")
        cursor.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'analysis'
            AND tablename = 'vehicle_detections';
        """)
        indexes = cursor.fetchall()
        print("\nIndexes:")
        print("-------------")
        for idx in indexes:
            print(f"Name: {idx['indexname']}")
            print(f"Definition: {idx['indexdef']}")
            print("-------------")

        # Check for materialized view
        logger.info("Checking materialized view...")
        cursor.execute("""
            SELECT matviewname, definition
            FROM pg_matviews
            WHERE schemaname = 'analysis'
            AND matviewname = 'mv_vehicle_statistics';
        """)
        matviews = cursor.fetchall()
        print("\nMaterialized Views:")
        print("-------------")
        for view in matviews:
            print(f"Name: {view['matviewname']}")
            print(f"Definition: {view['definition']}")
            print("-------------")

        # Check for functions
        logger.info("Checking functions...")
        cursor.execute("""
            SELECT proname, prosrc
            FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'analysis'
            AND proname LIKE '%vehicle%';
        """)
        functions = cursor.fetchall()
        print("\nFunctions:")
        print("-------------")
        for func in functions:
            print(f"Name: {func['proname']}")
            print("-------------")

        cursor.close()
        conn.close()
        
        return True

    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    success = verify_database_structure()
    if success:
        print("\nDatabase verification completed successfully!")
    else:
        print("\nDatabase verification failed!")