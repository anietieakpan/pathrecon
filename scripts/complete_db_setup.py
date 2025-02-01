# scripts/complete_db_setup.py

import psycopg2
from config import Config
import logging

logger = logging.getLogger(__name__)

def create_missing_components():
    """Create missing database components"""
    try:
        conn = psycopg2.connect(
            dbname=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD,
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Create materialized view
        logger.info("Creating materialized view...")
        cursor.execute("""
            DROP MATERIALIZED VIEW IF EXISTS analysis.mv_vehicle_statistics CASCADE;
            
            CREATE MATERIALIZED VIEW analysis.mv_vehicle_statistics AS
            SELECT 
                vehicle_make,
                vehicle_model,
                vehicle_color,
                vehicle_type,
                COUNT(*) as detection_count,
                COUNT(DISTINCT plate_text) as unique_vehicles,
                MIN(timestamp_utc) as first_seen,
                MAX(timestamp_utc) as last_seen,
                jsonb_agg(DISTINCT vehicle_confidence_scores) as confidence_distribution
            FROM analysis.vehicle_detections
            WHERE vehicle_make IS NOT NULL
            GROUP BY vehicle_make, vehicle_model, vehicle_color, vehicle_type;

            CREATE UNIQUE INDEX idx_vehicle_statistics 
            ON analysis.mv_vehicle_statistics (vehicle_make, vehicle_model, vehicle_color, vehicle_type)
            WHERE vehicle_make IS NOT NULL;
        """)
        logger.info("Materialized view created successfully")

        # Create statistics refresh function
        logger.info("Creating refresh function...")
        cursor.execute("""
            CREATE OR REPLACE FUNCTION analysis.refresh_vehicle_statistics()
            RETURNS void AS $$
            BEGIN
                REFRESH MATERIALIZED VIEW CONCURRENTLY analysis.mv_vehicle_statistics;
            END;
            $$ LANGUAGE plpgsql;
        """)

        # Create trigger for automatic refresh
        logger.info("Creating refresh trigger...")
        cursor.execute("""
            CREATE OR REPLACE FUNCTION analysis.trigger_refresh_vehicle_stats()
            RETURNS TRIGGER AS $$
            BEGIN
                -- Only refresh if enough changes have accumulated
                IF (TG_OP = 'INSERT' AND (NEW.vehicle_make IS NOT NULL)) OR
                   (TG_OP = 'UPDATE' AND (NEW.vehicle_make IS NOT NULL OR OLD.vehicle_make IS NOT NULL)) THEN
                    PERFORM analysis.refresh_vehicle_statistics();
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            DROP TRIGGER IF EXISTS vehicle_stats_refresh_trigger 
            ON analysis.vehicle_detections;

            CREATE TRIGGER vehicle_stats_refresh_trigger
            AFTER INSERT OR UPDATE
            ON analysis.vehicle_detections
            FOR EACH ROW
            EXECUTE FUNCTION analysis.trigger_refresh_vehicle_stats();
        """)

        # Create vehicle data cleanup function
        logger.info("Creating cleanup function...")
        cursor.execute("""
            CREATE OR REPLACE FUNCTION analysis.cleanup_vehicle_data(
                retention_days INTEGER DEFAULT 90
            )
            RETURNS void AS $$
            BEGIN
                DELETE FROM analysis.vehicle_detections
                WHERE timestamp_utc < NOW() - (retention_days || ' days')::INTERVAL;
                
                PERFORM analysis.refresh_vehicle_statistics();
            END;
            $$ LANGUAGE plpgsql;
        """)

        logger.info("All components created successfully!")
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Error creating components: {str(e)}")
        return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if create_missing_components():
        print("\nDatabase setup completed successfully!")
        print("You can now run verify_db.py again to confirm all components are present.")
    else:
        print("\nDatabase setup failed!")