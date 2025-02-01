# app/database/analysis/schema.py

from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import CreateTable
import logging
from datetime import datetime, timedelta

Base = declarative_base()

class VehicleDetection(Base):
    __tablename__ = 'vehicle_detections'
    __table_args__ = {
        'postgresql_partition_by': 'RANGE (timestamp_utc)',
        'info': {'partition_by': 'RANGE (timestamp_utc)'}
    }
    
    id = Column(Integer, primary_key=True)
    plate_text = Column(String, index=True)
    confidence = Column(Float)
    timestamp_utc = Column(DateTime(timezone=True), index=True)
    timestamp_local = Column(DateTime(timezone=True))
    camera_id = Column(String)
    location = Column(JSONB)  # Stores GPS coordinates and location metadata
    metadata = Column(JSONB)  # Flexible storage for additional detection data

class FollowingPattern(Base):
    __tablename__ = 'following_patterns'
    
    id = Column(Integer, primary_key=True)
    subject_plate = Column(String, index=True)
    follower_plate = Column(String, index=True)
    pattern_start = Column(DateTime(timezone=True), index=True)
    pattern_end = Column(DateTime(timezone=True), index=True)
    confidence_score = Column(Float)
    detection_count = Column(Integer)
    avg_time_gap = Column(Float)  # in seconds
    pattern_metadata = Column(JSONB)

class PatternAnalysisResult(Base):
    __tablename__ = 'pattern_analysis_results'
    
    id = Column(Integer, primary_key=True)
    analysis_timestamp = Column(DateTime(timezone=True), index=True)
    analysis_type = Column(String, index=True)
    result_data = Column(JSONB)
    confidence = Column(Float)

def create_partition_for_month(engine, table_name: str, start_date: datetime):
    """Create a new partition for a specific month"""
    end_date = (start_date.replace(day=1) + timedelta(days=32)).replace(day=1)
    partition_name = f"{table_name}_y{start_date.year}m{start_date.month:02d}"
    
    sql = f"""
    CREATE TABLE IF NOT EXISTS {partition_name}
    PARTITION OF {table_name}
    FOR VALUES FROM ('{start_date.isoformat()}') TO ('{end_date.isoformat()}');
    """
    
    with engine.connect() as conn:
        conn.execute(text(sql))

def setup_database(connection_string: str):
    """Initialize the analysis database schema and partitions"""
    engine = create_engine(connection_string)
    logger = logging.getLogger(__name__)
    
    try:
        # Create extensions
        with engine.connect() as conn:
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "btree_gist";'))
        
        # Create tables
        Base.metadata.create_all(engine)
        
        # Create initial partitions for vehicle_detections
        current_date = datetime.utcnow()
        for i in range(-2, 3):  # Create partitions for previous 2 months and next 2 months
            partition_date = (current_date.replace(day=1) + timedelta(days=32*i)).replace(day=1)
            create_partition_for_month(engine, 'vehicle_detections', partition_date)
        
        # Create indexes
        with engine.connect() as conn:
            # Indexes for vehicle_detections
            conn.execute(text('''
                CREATE INDEX IF NOT EXISTS idx_vehicle_detections_plate_time 
                ON vehicle_detections (plate_text, timestamp_utc);
            '''))
            
            # Indexes for following_patterns
            conn.execute(text('''
                CREATE INDEX IF NOT EXISTS idx_following_patterns_plates 
                ON following_patterns (subject_plate, follower_plate);
                
                CREATE INDEX IF NOT EXISTS idx_following_patterns_time 
                ON following_patterns USING gist (
                    tstzrange(pattern_start, pattern_end)
                );
            '''))
        
        # Create materialized view for pattern summary
        with engine.connect() as conn:
            conn.execute(text('''
                CREATE MATERIALIZED VIEW IF NOT EXISTS mv_pattern_summary AS
                SELECT 
                    subject_plate,
                    follower_plate,
                    COUNT(*) as pattern_count,
                    AVG(confidence_score) as avg_confidence,
                    MAX(pattern_end) as last_seen,
                    AVG(detection_count) as avg_detections
                FROM following_patterns
                WHERE pattern_end >= NOW() - INTERVAL '30 days'
                GROUP BY subject_plate, follower_plate;
                
                CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_pattern_summary 
                ON mv_pattern_summary (subject_plate, follower_plate);
            '''))
        
        # Create functions for pattern analysis
        with engine.connect() as conn:
            # Function to detect following patterns
            conn.execute(text('''
                CREATE OR REPLACE FUNCTION analyze_following_pattern(
                    p_subject_plate text,
                    p_follower_plate text,
                    p_time_window interval,
                    p_min_detections int,
                    p_max_gap interval
                ) RETURNS TABLE (
                    match_confidence float,
                    detection_sequence jsonb,
                    avg_time_gap interval
                ) AS $$
                BEGIN
                    RETURN QUERY
                    WITH detection_pairs AS (
                        SELECT 
                            s.timestamp_utc as subject_time,
                            f.timestamp_utc as follower_time,
                            f.timestamp_utc - s.timestamp_utc as time_gap
                        FROM vehicle_detections s
                        JOIN vehicle_detections f 
                            ON f.timestamp_utc > s.timestamp_utc 
                            AND f.timestamp_utc <= s.timestamp_utc + p_max_gap
                        WHERE s.plate_text = p_subject_plate 
                            AND f.plate_text = p_follower_plate
                            AND s.timestamp_utc >= NOW() - p_time_window
                    )
                    SELECT
                        LEAST(1.0, COUNT(*)::float / p_min_detections) as match_confidence,
                        jsonb_agg(jsonb_build_object(
                            'subject_time', subject_time,
                            'follower_time', follower_time,
                            'gap', time_gap
                        )) as detection_sequence,
                        AVG(time_gap) as avg_time_gap
                    FROM detection_pairs;
                END;
                $$ LANGUAGE plpgsql;
            '''))
            
            # Function to refresh pattern analysis
            conn.execute(text('''
                CREATE OR REPLACE FUNCTION refresh_pattern_analysis() RETURNS void AS $$
                BEGIN
                    -- Refresh materialized view
                    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_pattern_summary;
                    
                    -- Clean up old patterns
                    DELETE FROM following_patterns 
                    WHERE pattern_end < NOW() - INTERVAL '90 days';
                    
                    -- Clean up old analysis results
                    DELETE FROM pattern_analysis_results 
                    WHERE analysis_timestamp < NOW() - INTERVAL '90 days';
                END;
                $$ LANGUAGE plpgsql;
            '''))
        
        logger.info("Successfully initialized analysis database schema")
        
    except Exception as e:
        logger.error(f"Error setting up analysis database: {str(e)}")
        raise