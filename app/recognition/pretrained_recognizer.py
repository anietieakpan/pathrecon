# app/recognition/pretrained_recognizer.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path
import json
from . import VehicleAttributeRecognizer, VehicleAttributes

logger = logging.getLogger(__name__)

class PreTrainedVehicleRecognizer(VehicleAttributeRecognizer):
    """Enhanced vehicle attribute recognition using computer vision techniques"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize paths and mappings
        self.model_dir = Path('app/models/vehicle')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default mappings first
        self._init_default_mappings()
        
        # Try to load mappings from file (will override defaults if exists)
        self.load_class_mappings()
        
        # Initialize preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Vehicle recognizer initialized successfully")

        
        
        
    def _init_default_mappings(self):
        """Initialize default mappings in case file loading fails"""
        self.make_mapping = {
            "0": "BMW",
            "1": "Mercedes-Benz",
            "2": "Audi",
            "3": "Toyota",
            "4": "Honda",
            "5": "Volkswagen",
            "6": "Ford",
            "7": "Nissan",
            "8": "Porsche"
        }
        
        self.model_mapping = {
            "BMW": {
                "0": "3 Series",
                "1": "5 Series",
                "2": "7 Series",
                "3": "X5",
                "4": "X3",
                "5": "M3",
                "6": "M5",
                "7": "4 Series"
            },
            "Mercedes-Benz": {
                "0": "C-Class",
                "1": "E-Class",
                "2": "S-Class",
                "3": "GLE",
                "4": "GLC",
                "5": "A-Class"
            },
            "Audi": {
                "0": "A4",
                "1": "A6",
                "2": "Q5",
                "3": "Q7",
                "4": "RS6",
                "5": "S4"
            }
        }
        
        self.type_mapping = {
            "0": "Sedan",
            "1": "SUV",
            "2": "Coupe",
            "3": "Convertible",
            "4": "Wagon",
            "5": "Sports Car",
            "6": "Hatchback"
        }
        
        self.color_mapping = {
            "0": "Black",
            "1": "White",
            "2": "Silver",
            "3": "Gray",
            "4": "Dark Gray",
            "5": "Red",
            "6": "Blue",
            "7": "Dark Blue",
            "8": "Green",
            "9": "Brown",
            "10": "Gold"
        }
        
        logger.info("Default mappings initialized")

    def load_class_mappings(self):
        """Load vehicle class mappings from file"""
        try:
            mapping_file = self.model_dir / 'vehicle_mappings.json'
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    mappings = json.load(f)
                    # Update mappings if they exist in file
                    if 'makes' in mappings:
                        self.make_mapping = mappings['makes']
                    if 'models' in mappings:
                        self.model_mapping = mappings['models']
                    if 'body_types' in mappings:
                        self.type_mapping = mappings['body_types']
                    if 'color_mapping' in mappings:
                        self.color_mapping = mappings['color_mapping']
                    logger.info("Successfully loaded mappings from file")
            else:
                logger.warning(f"No mapping file found at {mapping_file}, using default mappings")
        except Exception as e:
            logger.error(f"Error loading mappings from file: {str(e)}")
            logger.warning("Using default mappings")
            
            
    def recognize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> VehicleAttributes:
        """Recognize vehicle attributes using multiple methods"""
        try:
            # Validate input
            if image is None or len(image.shape) != 3:
                raise ValueError("Invalid image input")
                
            # Crop vehicle region
            x1, y1, x2, y2 = bbox
            vehicle_crop = image[y1:y2, x1:x2]
            
            # Get color
            color, color_conf = self._analyze_color(vehicle_crop)
            
            # Analyze vehicle features
            make, model, make_conf = self._analyze_vehicle_features(vehicle_crop)
            
            # Get vehicle type
            vehicle_type = self._detect_vehicle_type(vehicle_crop)
            
            # Save vehicle image
            image_path = self._save_vehicle_image(vehicle_crop)
            
            return VehicleAttributes(
                make=make,
                model=model,
                color=color,
                year=None,
                type=vehicle_type,
                confidence_scores={
                    'make': float(make_conf),
                    'model': float(make_conf * 0.9),
                    'color': float(color_conf),
                    'type': 0.7
                },
                image_path=str(image_path) if image_path else None
            )
            
        except Exception as e:
            logger.error(f"Error in vehicle recognition: {str(e)}")
            return self._get_fallback_result()

    def _analyze_vehicle_features(self, image: np.ndarray) -> Tuple[str, str, float]:
        """Analyze vehicle features to determine make and model"""
        try:
            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # # Analyze front/rear view
            # is_front_view = self._is_front_view(edges)
            
            # if is_front_view:
            #     # Front grille analysis
            #     make, conf = self._analyze_front_grille(edges, gray)
            # else:
            #     # Side profile analysis
            #     make, conf = self._analyze_side_profile(edges, gray)
                
            # If make is detected, determine model

            make, conf = self._analyze_front_grille(edges, gray)
            
            if make != "Unknown":
                model = self._determine_model(make, image, edges)
            else:
                model = "Unknown"
                
            return make, model, conf
            
        except Exception as e:
            logger.error(f"Error analyzing vehicle features: {str(e)}")
            return "Unknown", "Unknown", 0.0
        
        
    def _analyze_color(self, image: np.ndarray) -> Tuple[str, float]:
        """Analyze vehicle color using multiple color spaces"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Get average colors in different spaces
            mean_bgr = np.mean(image, axis=(0, 1))
            mean_hsv = np.mean(hsv, axis=(0, 1))
            
            # Calculate brightness from LAB space
            brightness = np.mean(lab[:,:,0])
            
            # Color detection logic
            if brightness < 50:  # Dark
                if np.max(mean_bgr) < 60:
                    return "Black", 0.9
                elif mean_bgr[0] > mean_bgr[1] and mean_bgr[0] > mean_bgr[2]:
                    return "Dark Blue", 0.8
                else:
                    return "Black", 0.7
                    
            elif brightness > 200:  # Light
                if np.min(mean_bgr) > 200:
                    return "White", 0.9
                elif np.mean(mean_bgr) > 180:
                    return "Silver", 0.8
                else:
                    return "White", 0.7
                    
            else:  # Mid tones
                if mean_bgr[2] > 1.5 * (mean_bgr[0] + mean_bgr[1]) / 2:
                    return "Red", 0.85
                elif mean_bgr[0] > 1.5 * (mean_bgr[1] + mean_bgr[2]) / 2:
                    return "Blue", 0.85
                elif np.std(mean_bgr) < 20:
                    return "Gray", 0.8
                else:
                    return "Silver", 0.6
                    
        except Exception as e:
            logger.error(f"Error in color analysis: {str(e)}")
            return "Unknown", 0.0

    def _is_front_view(self, edges: np.ndarray) -> bool:
        """Determine if image shows front of vehicle"""
        try:
            height, width = edges.shape
            
            # Check symmetry
            left_half = edges[:, :width//2]
            right_half = cv2.flip(edges[:, width//2:], 1)
            
            # Calculate similarity between halves
            similarity = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)
            
            return similarity[0][0] > 0.5
        except Exception as e:
            logger.error(f"Error checking view: {str(e)}")
            return False

    def _analyze_front_grille(self, edges: np.ndarray, gray: np.ndarray) -> Tuple[str, float]:
        """Analyze front grille pattern for make identification"""
        try:
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Feature scores for each make
            scores = {
                'BMW': self._check_bmw_features(contours, gray),
                'Mercedes-Benz': self._check_mercedes_features(contours, gray),
                'Audi': self._check_audi_features(contours, gray)
            }
            
            # Get best match
            make, score = max(scores.items(), key=lambda x: x[1])
            
            # Return if confidence is high enough
            if score > 0.6:
                return make, score
                
            return "Unknown", 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing grille: {str(e)}")
            return "Unknown", 0.0

    def _check_bmw_features(self, contours: List[np.ndarray], gray: np.ndarray) -> float:
        """Check for BMW-specific features"""
        try:
            score = 0.0
            
            # Check for kidney grille
            kidney_score = 0.0
            for i, contour1 in enumerate(contours[:-1]):
                for contour2 in contours[i+1:]:
                    # Compare shapes for kidney grille pattern
                    similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0)
                    if similarity < 0.1:  # Very similar shapes
                        x1, y1, w1, h1 = cv2.boundingRect(contour1)
                        x2, y2, w2, h2 = cv2.boundingRect(contour2)
                        
                        # Check if they're side by side
                        if abs(y1 - y2) < h1/2 and abs(w1 - w2) < w1/4:
                            kidney_score = max(kidney_score, 1.0 - similarity)
            
            score += kidney_score * 0.7
            
            # Check for logo
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=15, maxRadius=50)
            if circles is not None:
                score += 0.3
                
            return score
            
        except Exception as e:
            logger.error(f"Error checking BMW features: {str(e)}")
            return 0.0

    def get_confidence(self) -> float:
        """Get overall confidence of the recognition system"""
        return 1.0

    def supports_attribute(self, attribute: str) -> bool:
        """Check if attribute is supported"""
        return attribute.lower() in ['make', 'model', 'color', 'type']

    def _detect_vehicle_type(self, image: np.ndarray) -> str:
        """Detect vehicle type based on shape analysis"""
        try:
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            if aspect_ratio > 2.5:
                return "Sedan"
            elif aspect_ratio < 1.8:
                return "SUV"
            elif aspect_ratio > 2.2:
                return "Sports Car"
            else:
                return "Unknown"
                
        except Exception as e:
            logger.error(f"Error in vehicle type detection: {str(e)}")
            return "Unknown"