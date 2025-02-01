# scripts/migrations/schema_updates.py

class VehicleDetailsMigration:
    def add_vehicle_columns(self):
        """Add new vehicle-related columns"""
        try:
            logger.info("Adding vehicle detail columns...")
            self.cursor.execute("""
                ALTER TABLE analysis.vehicle_detections 
                ADD COLUMN IF NOT EXISTS vehicle_make VARCHAR(50),
                ADD COLUMN IF NOT EXISTS vehicle_model VARCHAR(50),
                ADD COLUMN IF NOT EXISTS vehicle_color VARCHAR(30),
                ADD COLUMN IF NOT EXISTS vehicle_year INTEGER,
                ADD COLUMN IF NOT EXISTS vehicle_type VARCHAR(30),
                ADD COLUMN IF NOT EXISTS vehicle_image_path TEXT,
                ADD COLUMN IF NOT EXISTS vehicle_confidence_scores JSONB;
            """)
            logger.info("Vehicle detail columns added successfully")
        except Exception as e:
            logger.error(f"Failed to add vehicle columns: {str(e)}")
            raise

    def create_indexes(self):
        """Create new indexes for vehicle details"""
        try:
            logger.info("Creating indexes for vehicle details...")
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vehicle_make_model 
                ON analysis.vehicle_detections (vehicle_make, vehicle_model);

                CREATE INDEX IF NOT EXISTS idx_vehicle_color 
                ON analysis.vehicle_detections (vehicle_color);

                CREATE INDEX IF NOT EXISTS idx_vehicle_type 
                ON analysis.vehicle_detections (vehicle_type);

                CREATE INDEX IF NOT EXISTS idx_vehicle_year 
                ON analysis.vehicle_detections (vehicle_year);

                CREATE INDEX IF NOT EXISTS idx_vehicle_details_combined 
                ON analysis.vehicle_detections 
                (vehicle_make, vehicle_model, vehicle_color, vehicle_type)
                WHERE vehicle_make IS NOT NULL;
            """)
            logger.info("Indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise

    def create_statistics_view(self):
        """Create materialized view for vehicle statistics"""
        try:
            logger.info("Creating vehicle statistics view...")
            self.cursor.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vehicle_statistics AS
                SELECT 
                    vehicle_make,
                    vehicle_model,
                    vehicle_color,
                    vehicle_type,
                    COUNT(*) as detection_count,
                    COUNT(DISTINCT plate_text) as unique_vehicles,
                    MIN(timestamp_utc) as first_seen,
                    MAX(timestamp_utc) as last_seen,
                    jsonb_object_agg(
                        plate_text, 
                        jsonb_build_object(
                            'confidence', confidence,
                            'detections', COUNT(*)
                        )
                    ) as plate_statistics
                FROM analysis.vehicle_detections
                WHERE vehicle_make IS NOT NULL
                GROUP BY 
                    vehicle_make, 
                    vehicle_model, 
                    vehicle_color, 
                    vehicle_type;

                CREATE UNIQUE INDEX IF NOT EXISTS idx_vehicle_statistics 
                ON mv_vehicle_statistics 
                (vehicle_make, vehicle_model, vehicle_color, vehicle_type)
                WHERE vehicle_make IS NOT NULL;
            """)
            logger.info("Statistics view created successfully")
        except Exception as e:
            logger.error(f"Failed to create statistics view: {str(e)}")
            raise