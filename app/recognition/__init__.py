# app/recognition/__init__.py

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cv2

@dataclass
class VehicleAttributes:
    """Data class for vehicle attributes"""
    make: str
    model: str
    color: str
    year: Optional[int]
    type: str
    confidence_scores: Dict[str, float]
    image_path: Optional[str] = None

class RecognitionType(Enum):
    # """Types of available recognition models"""
    # BASIC = "basic"           # Basic implementation
    # PRETRAINED = "pretrained" # New pre-trained implementation
    """Types of available recognition models"""
    BASIC = "basic"           # Basic implementation
    PRETRAINED = "pretrained" # New pre-trained implementation
    RESNET = "resnet"        # Optional ResNet implementation


class VehicleAttributeRecognizer(ABC):
    """Abstract base class for vehicle attribute recognition"""
    
    @abstractmethod
    def recognize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> VehicleAttributes:
        """Recognize vehicle attributes from image"""
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """Get overall confidence score"""
        pass
    
    @abstractmethod
    def supports_attribute(self, attribute: str) -> bool:
        """Check if recognizer supports specific attribute"""
        pass
        
    def _save_vehicle_image(self, vehicle_crop: np.ndarray) -> Optional[str]:
        """Save vehicle crop image"""
        try:
            import uuid
            from pathlib import Path
            
            # Create output directory if it doesn't exist
            output_dir = Path("data/vehicle_images")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            filename = output_dir / f"vehicle_{uuid.uuid4()}.jpg"
            
            # Save image
            cv2.imwrite(str(filename), vehicle_crop)
            
            return str(filename)
        except Exception as e:
            import logging
            logging.error(f"Error saving vehicle image: {str(e)}")
            return None

    def _get_fallback_result(self) -> VehicleAttributes:
        """Return fallback result when recognition fails"""
        return VehicleAttributes(
            make="Unknown",
            model="Unknown",
            color="Unknown",
            year=None,
            type="Unknown",
            confidence_scores={
                'make': 0.0,
                'model': 0.0,
                'color': 0.0,
                'type': 0.0
            },
            image_path=None
        )

class VehicleRecognizerFactory:
    """Factory for creating vehicle recognizers"""
    
    @staticmethod
    def create_recognizer(recognition_type: RecognitionType = RecognitionType.PRETRAINED) -> VehicleAttributeRecognizer:
        """Create appropriate recognizer based on type"""
        if recognition_type == RecognitionType.BASIC:
            from .basic_recognizer import BasicVehicleRecognizer
            return BasicVehicleRecognizer()
        elif recognition_type == RecognitionType.PRETRAINED:
            from .pretrained_recognizer import PreTrainedVehicleRecognizer
            return PreTrainedVehicleRecognizer()
        elif recognition_type == RecognitionType.RESNET:
            from .resnet_recognizer import ResNetVehicleRecognizer
            return ResNetVehicleRecognizer()
        else:
            raise ValueError(f"Unsupported recognition type: {recognition_type}")