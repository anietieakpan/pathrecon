# scripts/migrations/maintenance_functions.py

class VehicleDetailsMigration:
    def create_maintenance_functions(self):
        """Create database maintenance functions"""
        try:
            logger.info("Creating maintenance functions...")
            
            # Function to refresh statistics
            self.cursor.execute("""
                CREATE OR REPLACE FUNCTION refresh_vehicle_statistics()
                RETURNS void AS $$
                BEGIN
                    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vehicle_statistics;
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Function to clean old data
            self.cursor.execute("""
                CREATE OR REPLACE FUNCTION cleanup_old_vehicle_data(
                    retention_days INTEGER DEFAULT 90
                )
                RETURNS void AS $$
                BEGIN
                    -- Delete old detections
                    DELETE FROM analysis.vehicle_detections
                    WHERE timestamp_utc < NOW() - (retention_days || ' days')::INTERVAL;
                    
                    -- Refresh statistics after cleanup
                    PERFORM refresh_vehicle_statistics();
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Function to update vehicle confidence scores
            self.cursor.execute("""
                CREATE OR REPLACE FUNCTION update_vehicle_confidence(
                    p_detection_id BIGINT,
                    p_confidence_scores JSONB
                )
                RETURNS void AS $$
                BEGIN
                    UPDATE analysis.vehicle_detections
                    SET vehicle_confidence_scores = p_confidence_scores,
                        metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{last_confidence_update}',
                            to_jsonb(NOW())
                        )
                    WHERE id = p_detection_id;
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Create automated maintenance trigger
            self.cursor.execute("""
                CREATE OR REPLACE FUNCTION trigger_vehicle_maintenance()
                RETURNS TRIGGER AS $$
                BEGIN
                    -- Refresh statistics if enough changes have accumulated
                    IF (SELECT COUNT(*) FROM pg_stat_user_tables 
                        WHERE relname = 'vehicle_detections' 
                        AND n_mod_since_analyze > 1000) > 0 
                    THEN
                        PERFORM refresh_vehicle_statistics();
                    END IF;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;

                DROP TRIGGER IF EXISTS vehicle_maintenance_trigger 
                ON analysis.vehicle_detections;

                CREATE TRIGGER vehicle_maintenance_trigger
                AFTER INSERT OR UPDATE OR DELETE
                ON analysis.vehicle_detections
                FOR EACH STATEMENT
                EXECUTE FUNCTION trigger_vehicle_maintenance();
            """)

            logger.info("Maintenance functions created successfully")
        except Exception as e:
            logger.error(f"Failed to create maintenance functions: {str(e)}")
            raise

    def validate_migration(self):
        """Validate the migration was successful"""
        try:
            logger.info("Validating migration...")
            
            # Check new columns
            self.cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns
                WHERE table_schema = 'analysis' 
                AND table_name = 'vehicle_detections';
            """)
            columns = self.cursor.fetchall()
            required_columns = {
                'vehicle_make', 'vehicle_model', 'vehicle_color',
                'vehicle_year', 'vehicle_type', 'vehicle_image_path',
                'vehicle_confidence_scores'
            }
            existing_columns = {col[0] for col in columns}
            missing_columns = required_columns - existing_columns
            
            if missing_columns:
                raise Exception(f"Missing columns: {missing_columns}")

            # Check indexes
            self.cursor.execute("""
                SELECT indexname 
                FROM pg_indexes
                WHERE schemaname = 'analysis' 
                AND tablename = 'vehicle_detections';
            """)
            indexes = self.cursor.fetchall()
            required_indexes = {
                'idx_vehicle_make_model',
                'idx_vehicle_color',
                'idx_vehicle_type',
                'idx_vehicle_year'
            }
            existing_indexes = {idx[0] for idx in indexes}
            missing_indexes = required_indexes - existing_indexes
            
            if missing_indexes:
                raise Exception(f"Missing indexes: {missing_indexes}")

            logger.info("Migration validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Migration validation failed: {str(e)}")
            raise