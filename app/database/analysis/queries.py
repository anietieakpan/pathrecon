# app/database/analysis/queries.py

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.orm import Session

class PatternAnalysis:
    def __init__(self, session: Session):
        self.session = session

    def find_following_patterns(
        self,
        time_window: timedelta = timedelta(hours=1),
        min_detections: int = 3,
        max_gap: timedelta = timedelta(minutes=10),
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find vehicles that appear to be following each other"""
        
        query = text("""
            WITH recent_detections AS (
                SELECT DISTINCT plate_text
                FROM vehicle_detections
                WHERE timestamp_utc >= NOW() - :time_window
            ),
            pattern_analysis AS (
                SELECT 
                    r1.plate_text as subject_plate,
                    r2.plate_text as follower_plate,
                    analyze_following_pattern(
                        r1.plate_text,
                        r2.plate_text,
                        :time_window,
                        :min_detections,
                        :max_gap
                    ) as pattern_data
                FROM recent_detections r1
                CROSS JOIN recent_detections r2
                WHERE r1.plate_text != r2.plate_text
            )
            SELECT 
                subject_plate,
                follower_plate,
                (pattern_data).match_confidence as confidence,
                (pattern_data).detection_sequence as detections,
                (pattern_data).avg_time_gap as avg_gap
            FROM pattern_analysis
            WHERE (pattern_data).match_confidence >= :min_confidence
            ORDER BY (pattern_data).match_confidence DESC;
        """)
        
        result = self.session.execute(
            query,
            {
                'time_window': time_window,
                'min_detections': min_detections,
                'max_gap': max_gap,
                'min_confidence': min_confidence
            }
        )
        
        return [dict(row) for row in result]

    def get_vehicle_history(
        self,
        plate_text: str,
        time_window: timedelta = timedelta(days=1)
    ) -> Dict[str, Any]:
        """Get comprehensive history for a specific vehicle"""
        
        query = text("""
            WITH vehicle_stats AS (
                SELECT
                    COUNT(*) as detection_count,
                    MIN(timestamp_utc) as first_seen,
                    MAX(timestamp_utc) as last_seen,
                    jsonb_agg(
                        jsonb_build_object(
                            'timestamp', timestamp_utc,
                            'location', location,
                            'camera_id', camera_id
                        )
                        ORDER BY timestamp_utc DESC
                    ) as detections
                FROM vehicle_detections
                WHERE plate_text = :plate_text
                AND timestamp_utc >= NOW() - :time_window
            ),
            following_stats AS (
                SELECT
                    COUNT(DISTINCT follower_plate) as unique_followers,
                    jsonb_agg(
                        jsonb_build_object(
                            'follower', follower_plate,
                            'confidence', confidence_score,
                            'start_time', pattern_start,
                            'end_time', pattern_end
                        )
                    ) as following_patterns
                FROM following_patterns
                WHERE subject_plate = :plate_text
                AND pattern_end >= NOW() - :time_window
            ),
            being_followed_stats AS (
                SELECT
                    COUNT(DISTINCT subject_plate) as following_count,
                    jsonb_agg(
                        jsonb_build_object(
                            'subject', subject_plate,
                            'confidence', confidence_score,
                            'start_time', pattern_start,
                            'end_time', pattern_end
                        )
                    ) as followed_patterns
                FROM following_patterns
                WHERE follower_plate = :plate_text
                AND pattern_end >= NOW() - :time_window
            )
            SELECT
                v.detection_count,
                v.first_seen,
                v.last_seen,
                v.detections,
                f.unique_followers,
                f.following_patterns,
                bf.following_count,
                bf.followed_patterns
            FROM vehicle_stats v
            LEFT JOIN following_stats f ON true
            LEFT JOIN being_followed_stats bf ON true;
        """)
        
        result = self.session.execute(query, {
            'plate_text': plate_text,
            'time_window': time_window
        }).first()
        
        return dict(result) if result else None
    
    def find_common_locations(
        self,
        min_occurrences: int = 3,
        time_window: timedelta = timedelta(days=7)
    ) -> List[Dict[str, Any]]:
        """Find locations where vehicles are frequently detected"""
        
        query = text("""
            WITH location_counts AS (
                SELECT
                    location->>'area' as area,
                    location->>'coordinates' as coordinates,
                    COUNT(DISTINCT plate_text) as unique_vehicles,
                    COUNT(*) as total_detections,
                    jsonb_agg(DISTINCT plate_text) as plates_detected
                FROM vehicle_detections
                WHERE timestamp_utc >= NOW() - :time_window
                AND location IS NOT NULL
                GROUP BY location->>'area', location->>'coordinates'
                HAVING COUNT(*) >= :min_occurrences
            )
            SELECT
                area,
                coordinates,
                unique_vehicles,
                total_detections,
                plates_detected
            FROM location_counts
            ORDER BY total_detections DESC;
        """)
        
        result = self.session.execute(query, {
            'min_occurrences': min_occurrences,
            'time_window': time_window
        })
        
        return [dict(row) for row in result]
    
    
    def analyze_time_patterns(
        self,
        plate_text: str,
        time_window: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """Analyze temporal patterns for a specific vehicle"""
        
        query = text("""
            WITH hourly_patterns AS (
                SELECT
                    EXTRACT(HOUR FROM timestamp_local) as hour,
                    COUNT(*) as detection_count,
                    COUNT(DISTINCT DATE(timestamp_local)) as unique_days
                FROM vehicle_detections
                WHERE plate_text = :plate_text
                AND timestamp_utc >= NOW() - :time_window
                GROUP BY EXTRACT(HOUR FROM timestamp_local)
            ),
            daily_patterns AS (
                SELECT
                    EXTRACT(DOW FROM timestamp_local) as day_of_week,
                    COUNT(*) as detection_count,
                    COUNT(DISTINCT DATE(timestamp_local)) as unique_weeks
                FROM vehicle_detections
                WHERE plate_text = :plate_text
                AND timestamp_utc >= NOW() - :time_window
                GROUP BY EXTRACT(DOW FROM timestamp_local)
            ),
            location_patterns AS (
                SELECT
                    location->>'area' as area,
                    COUNT(*) as visit_count,
                    MIN(timestamp_utc) as first_visit,
                    MAX(timestamp_utc) as last_visit
                FROM vehicle_detections
                WHERE plate_text = :plate_text
                AND timestamp_utc >= NOW() - :time_window
                AND location IS NOT NULL
                GROUP BY location->>'area'
            )
            SELECT
                jsonb_build_object(
                    'hourly_patterns', (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'hour', hour,
                                'detection_count', detection_count,
                                'unique_days', unique_days
                            )
                            ORDER BY hour
                        )
                        FROM hourly_patterns
                    ),
                    'daily_patterns', (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'day_of_week', day_of_week,
                                'detection_count', detection_count,
                                'unique_weeks', unique_weeks
                            )
                            ORDER BY day_of_week
                        )
                        FROM daily_patterns
                    ),
                    'location_patterns', (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'area', area,
                                'visit_count', visit_count,
                                'first_visit', first_visit,
                                'last_visit', last_visit
                            )
                            ORDER BY visit_count DESC
                        )
                        FROM location_patterns
                    )
                ) as patterns;
        """)
        
        result = self.session.execute(query, {
            'plate_text': plate_text,
            'time_window': time_window
        }).first()
        
        return result[0] if result else None