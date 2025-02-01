# app/database/analysis/examples.py

from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any
import logging
from .queries import PatternAnalysis

logger = logging.getLogger(__name__)

class VehicleAnalysisService:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        
    def analyze_potential_following(self, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """
        Analyze recent vehicle detections for potential following patterns.
        
        Example usage:
        ```python
        analysis_service = VehicleAnalysisService("postgresql://user:pass@localhost/analysis_db")
        following_patterns = analysis_service.analyze_potential_following(min_confidence=0.8)
        for pattern in following_patterns:
            print(f"Vehicle {pattern['subject_plate']} potentially being followed by {pattern['follower_plate']}")
            print(f"Confidence: {pattern['confidence']:.2f}")
        ```
        """
        try:
            session = self.Session()
            analyzer = PatternAnalysis(session)
            
            # Look for patterns in the last hour
            patterns = analyzer.find_following_patterns(
                time_window=timedelta(hours=1),
                min_detections=3,
                max_gap=timedelta(minutes=5),
                min_confidence=min_confidence
            )
            
            return patterns
        except Exception as e:
            logger.error(f"Error analyzing following patterns: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_vehicle_profile(self, plate_text: str) -> Dict[str, Any]:
        """
        Get comprehensive profile of a vehicle including its history and patterns.
        
        Example usage:
        ```python
        profile = analysis_service.get_vehicle_profile("ABC123")
        if profile['is_frequent']:
            print(f"Frequent visitor detected: {profile['total_visits']} visits")
            print(f"Common locations: {', '.join(profile['common_locations'])}")
        ```
        """
        try:
            session = self.Session()
            analyzer = PatternAnalysis(session)
            
            # Get vehicle history
            history = analyzer.get_vehicle_history(
                plate_text=plate_text,
                time_window=timedelta(days=30)
            )
            
            # Get time patterns
            time_patterns = analyzer.analyze_time_patterns(
                plate_text=plate_text,
                time_window=timedelta(days=30)
            )
            
            if not history:
                return {
                    'plate_text': plate_text,
                    'is_frequent': False,
                    'total_visits': 0,
                    'profile_available': False
                }
            
            # Analyze the data to create a profile
            profile = {
                'plate_text': plate_text,
                'is_frequent': history['detection_count'] > 10,
                'total_visits': history['detection_count'],
                'profile_available': True,
                'first_seen': history['first_seen'],
                'last_seen': history['last_seen'],
                'following_patterns': {
                    'as_subject': len(history.get('following_patterns', [])),
                    'as_follower': len(history.get('followed_patterns', []))
                },
                'temporal_patterns': self._analyze_temporal_patterns(time_patterns),
                'risk_assessment': self._assess_risk(history, time_patterns)
            }
            
            return profile
        except Exception as e:
            logger.error(f"Error getting vehicle profile: {str(e)}")
            raise
        finally:
            session.close()
    
    def find_location_hotspots(self) -> List[Dict[str, Any]]:
        """
        Identify locations with unusual vehicle activity patterns.
        
        Example usage:
        ```python
        hotspots = analysis_service.find_location_hotspots()
        for hotspot in hotspots:
            print(f"Hotspot detected at {hotspot['area']}")
            print(f"Unique vehicles: {hotspot['unique_vehicles']}")
            print(f"Activity level: {hotspot['activity_level']}")
        ```
        """
        try:
            session = self.Session()
            analyzer = PatternAnalysis(session)
            
            locations = analyzer.find_common_locations(
                min_occurrences=5,
                time_window=timedelta(days=7)
            )
            
            # Analyze locations for patterns
            hotspots = []
            for loc in locations:
                activity_level = self._calculate_activity_level(loc)
                if activity_level > 0.7:  # High activity threshold
                    hotspots.append({
                        'area': loc['area'],
                        'coordinates': loc['coordinates'],
                        'unique_vehicles': loc['unique_vehicles'],
                        'total_detections': loc['total_detections'],
                        'activity_level': activity_level,
                        'significance': self._assess_location_significance(loc)
                    })
            
            return sorted(hotspots, key=lambda x: x['activity_level'], reverse=True)
        except Exception as e:
            logger.error(f"Error finding location hotspots: {str(e)}")
            raise
        finally:
            session.close()

    def _analyze_temporal_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns to identify significant timing patterns"""
        if not patterns:
            return {}
            
        hourly = patterns.get('hourly_patterns', [])
        daily = patterns.get('daily_patterns', [])
        
        return {
            'peak_hours': self._find_peak_hours(hourly),
            'active_days': self._find_active_days(daily),
            'regularity_score': self._calculate_regularity(hourly, daily)
        }
    
    def _assess_risk(self, history: Dict[str, Any], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential risk factors based on vehicle patterns"""
        risk_factors = []
        risk_score = 0.0
        
        # Check for following patterns
        if history.get('unique_followers', 0) > 2:
            risk_factors.append('multiple_followers')
            risk_score += 0.3
            
        # Check for unusual timing patterns
        if patterns and self._has_unusual_timing(patterns):
            risk_factors.append('unusual_timing')
            risk_score += 0.2
            
        # Check for location patterns
        if self._has_suspicious_location_pattern(history):
            risk_factors.append('suspicious_locations')
            risk_score += 0.25
            
        return {
            'risk_score': min(1.0, risk_score),
            'risk_factors': risk_factors,
            'confidence': self._calculate_risk_confidence(history)
        }
            
    # Helper methods (implement based on your specific needs)
    def _calculate_activity_level(self, location_data: Dict[str, Any]) -> float:
        # Implementation details...
        pass
    
    def _assess_location_significance(self, location_data: Dict[str, Any]) -> str:
        # Implementation details...
        pass
    
    def _find_peak_hours(self, hourly_patterns: List[Dict[str, Any]]) -> List[int]:
        # Implementation details...
        pass
    
    def _find_active_days(self, daily_patterns: List[Dict[str, Any]]) -> List[int]:
        # Implementation details...
        pass
    
    def _calculate_regularity(self, hourly: List[Dict[str, Any]], 
                            daily: List[Dict[str, Any]]) -> float:
        # Implementation details...
        pass