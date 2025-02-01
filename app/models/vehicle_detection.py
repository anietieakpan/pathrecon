# app/models/vehicle_detection.py

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class VehicleDetails:
    make: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    year: Optional[int] = None
    type: Optional[str] = None  # sedan, suv, truck etc
    image_path: Optional[str] = None
    confidence_scores: Dict[str, float] = None

@dataclass
class VehicleDetection:
    plate_text: str
    confidence: float
    timestamp_utc: datetime
    timestamp_local: datetime
    vehicle_details: Optional[VehicleDetails] = None
    location: Optional[Dict[str, Any]] = None
    camera_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def has_vehicle_details(self) -> bool:
        return self.vehicle_details is not None

class VehicleClassifier:
    """Interface for vehicle classification services"""
    
    def classify_vehicle(self, image) -> VehicleDetails:
        """Classify vehicle details from image"""
        raise NotImplementedError
        
    def validate_results(self, details: VehicleDetails) -> bool:
        """Validate classification results"""
        raise NotImplementedError