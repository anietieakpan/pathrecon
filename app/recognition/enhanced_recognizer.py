# app/recognition/enhanced_recognizer.py

import torch
import torchvision
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from PIL import Image

from .base import VehicleAttributeRecognizer, VehicleAttributes
from ..utils.image_processing import preprocess_image, apply_roi_crop

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    weights_path: str
    input_size: Tuple[int, int]
    num_classes: int
    class_mapping: Dict[int, str]

class EnhancedVehicleRecognizer(VehicleAttributeRecognizer):
    """Enhanced vehicle attribute recognition using multiple specialized models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.transforms = {}
        
        # Initialize all models
        self._init_make_model_detector()
        self._init_color_detector()
        self._init_type_detector()
        self._init_year_estimator()
        
        logger.info(f"Enhanced Vehicle Recognizer initialized on {self.device}")

    def _init_make_model_detector(self):
        """Initialize make/model detection model (EfficientNet-based)"""
        model = torchvision.models.efficientnet_v2_l(weights=None)
        num_classes = 1000  # Number of make/model combinations
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        
        weights_path = Path("app/models/vehicle/make_model_classifier.pth")
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        
        self.models['make_model'] = model
        self.transforms['make_model'] = torchvision.transforms.Compose([
            torchvision.transforms.Resize((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
        ])

    def _init_color_detector(self):
        """Initialize color detection model (Custom CNN)"""
        model = torchvision.models.resnet18(weights=None)
        num_colors = 12  # Common vehicle colors
        model.fc = torch.nn.Linear(model.fc.in_features, num_colors)
        
        weights_path = Path("app/models/vehicle/color_classifier.pth")
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        
        self.models['color'] = model
        self.transforms['color'] = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
        ])

    def _init_type_detector(self):
        """Initialize vehicle type detection model (Modified ResNet)"""
        model = torchvision.models.resnet34(weights=None)
        num_types = 8  # sedan, suv, truck, etc.
        model.fc = torch.nn.Linear(model.fc.in_features, num_types)
        
        weights_path = Path("app/models/vehicle/type_classifier.pth")
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            
        model = model.to(self.device)
        model.eval()
        
        self.models['type'] = model
        self.transforms['type'] = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
        ])

    def _init_year_estimator(self):
        """Initialize vehicle year estimation model (Modified EfficientNet)"""
        model = torchvision.models.efficientnet_b0(weights=None)
        # Regression output for year
        model.classifier[1] = torch.nn.Sequential(
            torch.nn.Linear(model.classifier[1].in_features, 1),
            torch.nn.ReLU()  # Ensure positive output
        )
        
        weights_path = Path("app/models/vehicle/year_estimator.pth")
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            
        model = model.to(self.device)
        model.eval()
        
        self.models['year'] = model
        self.transforms['year'] = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def recognize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> VehicleAttributes:
        """Recognize all vehicle attributes using ensemble of models"""
        try:
            # Crop and preprocess ROI
            roi = apply_roi_crop(image, bbox)
            pil_image = Image.fromarray(roi)
            
            # Get make/model
            make, model_name, make_model_conf = self._detect_make_model(pil_image)
            
            # Get color
            color, color_conf = self._detect_color(pil_image)
            
            # Get vehicle type
            vehicle_type, type_conf = self._detect_vehicle_type(pil_image)
            
            # Estimate year
            year, year_conf = self._estimate_year(pil_image)
            
            # Save vehicle image
            image_path = self._save_vehicle_image(roi)
            
            return VehicleAttributes(
                make=make,
                model=model_name,
                color=color,
                year=year,
                type=vehicle_type,
                confidence_scores={
                    'make': float(make_model_conf),
                    'model': float(make_model_conf * 0.9),  # Slightly lower confidence for specific model
                    'color': float(color_conf),
                    'type': float(type_conf),
                    'year': float(year_conf)
                },
                image_path=str(image_path) if image_path else None
            )
            
        except Exception as e:
            logger.error(f"Error in vehicle recognition: {str(e)}")
            return self._get_fallback_result()

    def _detect_make_model(self, image: Image) -> Tuple[str, str, float]:
        """Detect vehicle make and model"""
        try:
            input_tensor = self.transforms['make_model'](image).unsqueeze(0).to(self.device)
            outputs = self.models['make_model'](input_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            conf, pred = torch.max(probs, 1)
            make_model = self._get_make_model_mapping(pred.item())
            
            return make_model['make'], make_model['model'], conf.item()
        except Exception as e:
            logger.error(f"Make/model detection failed: {str(e)}")
            return "Unknown", "Unknown", 0.0

    def _detect_color(self, image: Image) -> Tuple[str, float]:
        """Detect vehicle color"""
        try:
            input_tensor = self.transforms['color'](image).unsqueeze(0).to(self.device)
            outputs = self.models['color'](input_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            conf, pred = torch.max(probs, 1)
            color = self._get_color_mapping(pred.item())
            
            return color, conf.item()
        except Exception as e:
            logger.error(f"Color detection failed: {str(e)}")
            return "Unknown", 0.0

    def _detect_vehicle_type(self, image: Image) -> Tuple[str, float]:
        """Detect vehicle type"""
        try:
            input_tensor = self.transforms['type'](image).unsqueeze(0).to(self.device)
            outputs = self.models['type'](input_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            conf, pred = torch.max(probs, 1)
            vehicle_type = self._get_type_mapping(pred.item())
            
            return vehicle_type, conf.item()
        except Exception as e:
            logger.error(f"Vehicle type detection failed: {str(e)}")
            return "Unknown", 0.0

    def _estimate_year(self, image: Image) -> Tuple[Optional[int], float]:
        """Estimate vehicle year"""
        try:
            input_tensor = self.transforms['year'](image).unsqueeze(0).to(self.device)
            output = self.models['year'](input_tensor)
            
            # Convert to year (e.g., 2000-2024 range)
            year = int(2000 + (output.item() * 24))  # Scale output to year range
            confidence = min(1.0, max(0.0, 1.0 - abs(output.item() - 0.5)))  # Higher confidence near middle of range
            
            return year, confidence
        except Exception as e:
            logger.error(f"Year estimation failed: {str(e)}")
            return None, 0.0

    @staticmethod
    def _get_make_model_mapping(index: int) -> Dict[str, str]:
        """Get make/model from class index"""
        # Load from JSON or hardcoded mapping
        # This should be replaced with actual mapping
        return {"make": "Unknown", "model": "Unknown"}

    @staticmethod
    def _get_color_mapping(index: int) -> str:
        """Get color name from class index"""
        colors = {
            0: "Black",
            1: "White",
            2: "Silver",
            3: "Gray",
            4: "Red",
            5: "Blue",
            6: "Green",
            7: "Brown",
            8: "Gold",
            9: "Orange",
            10: "Yellow",
            11: "Purple"
        }
        return colors.get(index, "Unknown")

    @staticmethod
    def _get_type_mapping(index: int) -> str:
        """Get vehicle type from class index"""
        types = {
            0: "Sedan",
            1: "SUV",
            2: "Truck",
            3: "Van",
            4: "Coupe",
            5: "Wagon",
            6: "Hatchback",
            7: "Convertible"
        }
        return types.get(index, "Unknown")

    def get_confidence(self) -> float:
        """Get overall confidence of the recognition system"""
        return 1.0 if all(model.training == False for model in self.models.values()) else 0.0

    def supports_attribute(self, attribute: str) -> bool:
        """Check if attribute is supported"""
        return attribute.lower() in ['make', 'model', 'color', 'year', 'type']