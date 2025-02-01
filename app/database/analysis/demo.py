# app/database/analysis/demo.py

from datetime import datetime, timedelta
import logging
from .examples import VehicleAnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_analysis_demo():
    """Demo showing how to use the VehicleAnalysisService"""
    
    # Initialize the service
    db_url = "postgresql://user:pass@localhost:5432/analysis_db"
    analysis_service = VehicleAnalysisService(db_url)
    
    try:
        # 1. Check for following patterns
        logger.info("Checking for following patterns...")
        following_patterns = analysis_service.analyze_potential_following(min_confidence=0.8)
        
        for pattern in following_patterns:
            logger.info(f"""
                Following Pattern Detected:
                Subject: {pattern['subject_plate']}
                Follower: {pattern['follower_plate']}
                Confidence: {pattern['confidence']:.2f}
                Average Gap: {pattern['avg_gap']} seconds
            """)
        
        # 2. Analyze specific vehicle
        target_plate = "ABC123"
        logger.info(f"\nAnalyzing vehicle profile for {target_plate}...")
        profile = analysis_service.get_vehicle_profile(target_plate)
        
        if profile['profile_available']:
            logger.info(f"""
                Vehicle Profile:
                Plate: {profile['plate_text']}
                Total Visits: {profile['total_visits']}
                First Seen: {profile['first_seen']}
                Last Seen: {profile['last_seen']}
                Risk Score: {profile['risk_assessment']['risk_score']:.2f}
                Risk Factors: {', '.join(profile['risk_assessment']['risk_factors'])}
            """)
            
            if profile['temporal_patterns']:
                logger.info(f"""
                    Temporal Patterns:
                    Peak Hours: {profile['temporal_patterns']['peak_hours']}
                    Active Days: {profile['temporal_patterns']['active_days']}
                    Regularity Score: {profile['temporal_patterns']['regularity_score']:.2f}
                """)
        
        # 3. Find location hotspots
        logger.info("\nFinding location hotspots...")
        hotspots = analysis_service.find_location_hotspots()
        
        for hotspot in hotspots:
            logger.info(f"""
                Hotspot Location:
                Area: {hotspot['area']}
                Activity Level: {hotspot['activity_level']:.2f}
                Unique Vehicles: {hotspot['unique_vehicles']}
                Significance: {hotspot['significance']}
            """)

    except Exception as e:
        logger.error(f"Error in analysis demo: {str(e)}")

if __name__ == "__main__":
    run_analysis_demo()