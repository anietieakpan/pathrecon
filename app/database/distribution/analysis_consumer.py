# app/database/distribution/analysis_consumer.py

from datetime import datetime, timedelta
from typing import List, Dict, Any, Set, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from .core import DataConsumer, DetectionEvent

Base = declarative_base()

class VehicleDetection(Base):
    """Represents a single vehicle detection"""
    __tablename__ = 'vehicle_detections'
    
    id = Column(Integer, primary_key=True)
    plate_text = Column(String, index=True)
    confidence = Column(Float)
    timestamp_utc = Column(DateTime, index=True)
    timestamp_local = Column(DateTime)
    camera_id = Column(String, nullable=True)
    
    # Relationships
    following_patterns = relationship('FollowingPattern', 
                                    foreign_keys='FollowingPattern.follower_detection_id')
    being_followed_patterns = relationship('FollowingPattern',
                                         foreign_keys='FollowingPattern.subject_detection_id')

class FollowingPattern(Base):
    """Represents a detected following pattern between vehicles"""
    __tablename__ = 'following_patterns'
    
    id = Column(Integer, primary_key=True)
    subject_detection_id = Column(Integer, ForeignKey('vehicle_detections.id'))
    follower_detection_id = Column(Integer, ForeignKey('vehicle_detections.id'))
    pattern_start = Column(DateTime)
    pattern_end = Column(DateTime)
    confidence_score = Column(Float)  # Confidence in the following pattern
    
    # Pattern metadata
    detection_count = Column(Integer)  # Number of times following pattern observed
    avg_time_gap = Column(Float)  # Average time between detections
    
    subject_detection = relationship('VehicleDetection', foreign_keys=[subject_detection_id])
    follower_detection = relationship('VehicleDetection', foreign_keys=[follower_detection_id])

class AnalysisDatabaseConsumer(DataConsumer):
    """Consumes detection events and analyzes for patterns"""
    
    def __init__(self, connection_string: str, 
                 pattern_window: timedelta = timedelta(hours=1),
                 min_detections: int = 3,
                 max_time_gap: timedelta = timedelta(minutes=10)):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.pattern_window = pattern_window
        self.min_detections = min_detections
        self.max_time_gap = max_time_gap
        self.last_processed_time = None

    def process_detection(self, detection: DetectionEvent) -> bool:
        """Process a single detection event"""
        session = self.Session()
        try:
            # Store detection
            db_detection = VehicleDetection(
                plate_text=detection.plate_text,
                confidence=detection.confidence,
                timestamp_utc=detection.timestamp_utc,
                timestamp_local=detection.timestamp_local,
                camera_id=detection.camera_id
            )
            session.add(db_detection)
            session.commit()
            
            # Analyze patterns
            self._analyze_following_patterns(session, db_detection)
            
            self.last_processed_time = detection.timestamp_utc
            return True
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def process_batch(self, detections: List[DetectionEvent]) -> bool:
        """Process a batch of detection events"""
        session = self.Session()
        try:
            # Store all detections
            db_detections = []
            for detection in detections:
                db_detection = VehicleDetection(
                    plate_text=detection.plate_text,
                    confidence=detection.confidence,
                    timestamp_utc=detection.timestamp_utc,
                    timestamp_local=detection.timestamp_local,
                    camera_id=detection.camera_id
                )
                session.add(db_detection)
                db_detections.append(db_detection)
            
            session.commit()
            
            # Analyze patterns for each detection
            for db_detection in db_detections:
                self._analyze_following_patterns(session, db_detection)
            
            if detections:
                self.last_processed_time = max(d.timestamp_utc for d in detections)
            return True
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _analyze_following_patterns(self, session, new_detection: VehicleDetection):
        """Analyze vehicle following patterns for a new detection"""
        # Get recent detections within pattern window
        window_start = new_detection.timestamp_utc - self.pattern_window
        recent_detections = session.query(VehicleDetection)\
            .filter(VehicleDetection.timestamp_utc >= window_start)\
            .filter(VehicleDetection.timestamp_utc <= new_detection.timestamp_utc)\
            .order_by(VehicleDetection.timestamp_utc)\
            .all()
        
        # Group detections by plate
        detections_by_plate = {}
        for detection in recent_detections:
            if detection.plate_text not in detections_by_plate:
                detections_by_plate[detection.plate_text] = []
            detections_by_plate[detection.plate_text].append(detection)
        
        # Analyze each vehicle's detections for following patterns
        for plate, detections in detections_by_plate.items():
            if plate == new_detection.plate_text:
                continue
                
            # Check if this vehicle could be following the new detection's vehicle
            self._check_following_pattern(session, new_detection, detections)
            
            # Check if the new detection's vehicle could be following this vehicle
            self._check_following_pattern(session, detections[-1], 
                                        detections_by_plate.get(new_detection.plate_text, []))

    def _check_following_pattern(self, session, subject: VehicleDetection, 
                               potential_follower_detections: List[VehicleDetection]):
        """Check if there's a following pattern between two vehicles"""
        if len(potential_follower_detections) < self.min_detections:
            return
            
        # Calculate time gaps between detections
        total_gap = timedelta()
        valid_sequence = True
        prev_time = None
        
        for detection in potential_follower_detections:
            if prev_time is not None:
                gap = detection.timestamp_utc - prev_time
                if gap > self.max_time_gap:
                    valid_sequence = False
                    break
                total_gap += gap
            prev_time = detection.timestamp_utc
        
        if not valid_sequence:
            return
            
        avg_gap = total_gap / (len(potential_follower_detections) - 1)
        
        # Calculate confidence score based on number of detections and time gaps
        confidence_score = min(1.0, len(potential_follower_detections) / self.min_detections)
        
        # Create or update following pattern
        pattern = session.query(FollowingPattern)\
            .filter_by(subject_detection_id=subject.id,
                      follower_detection_id=potential_follower_detections[-1].id)\
            .first()
            
        if pattern is None:
            pattern = FollowingPattern(
                subject_detection=subject,
                follower_detection=potential_follower_detections[-1],
                pattern_start=potential_follower_detections[0].timestamp_utc,
                pattern_end=potential_follower_detections[-1].timestamp_utc,
                confidence_score=confidence_score,
                detection_count=len(potential_follower_detections),
                avg_time_gap=avg_gap.total_seconds()
            )
            session.add(pattern)
        else:
            pattern.pattern_end = potential_follower_detections[-1].timestamp_utc
            pattern.confidence_score = confidence_score
            pattern.detection_count = len(potential_follower_detections)
            pattern.avg_time_gap = avg_gap.total_seconds()
            
        session.commit()

    def get_last_processed_time(self) -> datetime:
        """Get the timestamp of the last processed event"""
        return self.last_processed_time

    def get_following_patterns(self, min_confidence: float = 0.7,
                             time_window: timedelta = None) -> List[Dict[str, Any]]:
        """Get detected following patterns"""
        session = self.Session()
        try:
            query = session.query(FollowingPattern)\
                .filter(FollowingPattern.confidence_score >= min_confidence)
            
            if time_window:
                min_time = datetime.utcnow() - time_window
                query = query.filter(FollowingPattern.pattern_end >= min_time)
            
            patterns = query.all()
            
            return [{
                'subject_plate': pattern.subject_detection.plate_text,
                'follower_plate': pattern.follower_detection.plate_text,
                'start_time': pattern.pattern_start,
                'end_time': pattern.pattern_end,
                'confidence': pattern.confidence_score,
                'detections': pattern.detection_count,
                'avg_time_gap': pattern.avg_time_gap
            } for pattern in patterns]
            
        finally:
            session.close()