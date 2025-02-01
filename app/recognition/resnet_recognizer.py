# app/recognition/resnet_recognizer.py

import torch
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
from torchvision import models, transforms
from . import VehicleAttributeRecognizer, VehicleAttributes

logger = logging.getLogger(__name__)

class ResNetVehicleRecognizer(VehicleAttributeRecognizer):
    """ResNet-based vehicle attribute recognition"""
    
    def __init__(self):
        self.model_dir = Path('app/models/vehicle')
        self.data_dir = Path('app/data/vehicle')
        self.vehicle_images_path = Path("data/vehicle_images")
        self.vehicle_images_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.class_mapping = self._load_class_mapping()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def recognize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> VehicleAttributes:
        try:
            # Get vehicle crop
            x1, y1, x2, y2 = bbox
            vehicle_crop = image[y1:y2, x1:x2]
            
            # Save image
            image_path = self._save_vehicle_image(vehicle_crop)
            
            # Get all attributes
            make, model, year, make_conf = self._predict_vehicle(vehicle_crop)
            color, color_conf = self._detect_color(vehicle_crop)
            vehicle_type = self._get_vehicle_type(make, model)
            
            return VehicleAttributes(
                make=make,
                model=model,
                color=color,
                year=year,
                type=vehicle_type,
                confidence_scores={
                    'make': float(make_conf),
                    'model': float(make_conf * 0.9),
                    'color': float(color_conf),
                    'type': float(make_conf * 0.8)
                },
                image_path=str(image_path) if image_path else None
            )
            
        except Exception as e:
            logger.error(f"Error in ResNet recognition: {str(e)}")
            return self._get_fallback_result()

    def get_confidence(self) -> float:
        return 0.8 if self.model is not None else 0.0

    def supports_attribute(self, attribute: str) -> bool:
        return attribute.lower() in ['make', 'model', 'color', 'year', 'type']

    def _load_model(self):
        """Load ResNet model"""
        try:
            # Load pre-trained ResNet50
            model = models.resnet50(pretrained=True)
            
            # Modify for car classification
            num_classes = 196  # Stanford Cars Dataset
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            
            # Load weights if available
            weights_path = self.model_dir / 'resnet50_cars.pth'
            if weights_path.exists():
                model.load_state_dict(torch.load(weights_path, 
                                               map_location=self.device))
                logger.info("Loaded ResNet vehicle model")
            
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading ResNet model: {str(e)}")
            return None
            
    def _load_class_mapping(self) -> Dict:
        """Load vehicle class mapping"""
        try:
            mapping_path = self.data_dir / 'car_classes.json'
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading class mapping: {str(e)}")
            return {}
    
    # Rest of the methods (color detection, etc.) would go here...