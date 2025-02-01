# app/detection/vehicle_classifier.py

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging
from pathlib import Path
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class VehicleClassifier:
    """Enhanced service for classifying vehicle characteristics from images"""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.model_dir = Path('app/models/vehicle')
        self.data_dir = Path('app/data/vehicle')
        
        # Initialize model components
        self.net = None
        self.model_info = None
        self.class_mapping = None
        
        # Store detected vehicle crops
        self.vehicle_images_path = Path("data/vehicle_images")
        self.vehicle_images_path.mkdir(parents=True, exist_ok=True)
        
        # Load model and configuration
        try:
            self.load_model()
        except Exception as e:
            logger.warning(f"Failed to load vehicle classification model: {str(e)}")
            logger.warning("Vehicle classifier will run in fallback mode")

    
    
    def load_model(self):
        """Load model and class mappings with enhanced error handling"""
        try:
            # Check paths
            model_path = self.model_dir / 'googlenet_cars.caffemodel'
            prototxt_path = self.model_dir / 'deploy.prototxt'
        
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return
            if not prototxt_path.exists():
                logger.error(f"Prototxt file not found: {prototxt_path}")
                return
            
            # Load model
            self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
        
            # Verify model loaded successfully
            if self.net.empty():
                logger.error("Failed to load neural network model")
                return
            
            logger.info("Successfully loaded neural network model")
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.net = None
    
    
    
    
    

    def process_vehicle_image(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Process a vehicle image with enhanced color detection and classification"""
        try:
            x1, y1, x2, y2 = bbox
            vehicle_crop = image[y1:y2, x1:x2]
            
            # Save vehicle crop
            image_path = self._save_vehicle_image(vehicle_crop)
            
            # Detect color using enhanced method
            color, color_confidence = self._detect_color_enhanced(vehicle_crop)
            
            # Classify vehicle if model is available
            if self.net is not None and self.class_mapping is not None:
                # Prepare image for model
                blob = self._prepare_image(vehicle_crop)
                self.net.setInput(blob)
                
                # Get predictions
                predictions = self.net.forward()
                top_indices = predictions[0].argsort()[-5:][::-1]
                top_confidences = predictions[0][top_indices]
                
                # Get vehicle info for top prediction
                vehicle_info = self._get_vehicle_info(top_indices[0], top_confidences[0])
                vehicle_info['color'] = color
                vehicle_info['confidence_scores']['color'] = float(color_confidence)
                vehicle_info['image_path'] = str(image_path) if image_path else None
                
                return vehicle_info
            else:
                # Fallback to color-only detection
                return {
                    'make': 'Unknown',
                    'model': 'Unknown',
                    'color': color,
                    'type': 'Unknown',
                    'year': None,
                    'confidence_scores': {
                        'make': 0.0,
                        'model': 0.0,
                        'color': float(color_confidence),
                        'type': 0.0
                    },
                    'image_path': str(image_path) if image_path else None
                }
                
        except Exception as e:
            logger.error(f"Error processing vehicle image: {str(e)}")
            return self._get_fallback_prediction()

            
            

    def _detect_color_enhanced(self, image: np.ndarray) -> Tuple[str, float]:
        """Enhanced color detection using multiple color spaces"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Define comprehensive color ranges with improved values
            color_ranges = {
                'red': {
                    'hsv1': ([0, 50, 50], [10, 255, 255]),    # Lower red hue range
                    'hsv2': ([170, 50, 50], [180, 255, 255]), # Upper red hue range
                    'lab': ([20, 150, 150], [190, 255, 255])
                },
                'orange': {
                    'hsv': ([11, 50, 50], [25, 255, 255]),
                    'lab': ([50, 150, 150], [200, 255, 255])
                },
                'yellow': {
                    'hsv': ([26, 50, 50], [34, 255, 255])
                },
                'white': {
                    'hsv': ([0, 0, 200], [180, 30, 255]),
                    'lab': ([200, 110, 110], [255, 140, 140])
                },
                'black': {
                    'hsv': ([0, 0, 0], [180, 30, 50]),
                    'lab': ([0, 110, 110], [50, 140, 140])
                },
                'silver': {
                    'hsv': ([0, 0, 140], [180, 30, 200]),
                    'lab': ([150, 110, 110], [200, 140, 140])
                },
                'gray': {
                    'hsv': ([0, 0, 70], [180, 30, 140]),
                    'lab': ([100, 110, 110], [150, 140, 140])
                },
                'blue': {
                    'hsv': ([100, 50, 50], [130, 255, 255]),
                    'lab': ([0, 110, 150], [255, 130, 255])
                },
                'green': {
                    'hsv': ([35, 50, 50], [85, 255, 255])
                },
                'brown': {
                    'hsv': ([10, 50, 50], [20, 255, 255])
                }
            }
            
            # Calculate color scores
            height, width = image.shape[:2]
            total_pixels = height * width
            color_scores = {}
            
            for color_name, ranges in color_ranges.items():
                score = 0
                count = 0
                
                # Process HSV ranges
                if 'hsv' in ranges:
                    mask = cv2.inRange(hsv, np.array(ranges['hsv'][0]), 
                                    np.array(ranges['hsv'][1]))
                    score += cv2.countNonZero(mask)
                    count += 1
                
                # Process special case for red (wraps around hue)
                if 'hsv1' in ranges and 'hsv2' in ranges:
                    mask1 = cv2.inRange(hsv, np.array(ranges['hsv1'][0]), 
                                    np.array(ranges['hsv1'][1]))
                    mask2 = cv2.inRange(hsv, np.array(ranges['hsv2'][0]), 
                                    np.array(ranges['hsv2'][1]))
                    mask = cv2.bitwise_or(mask1, mask2)
                    score += cv2.countNonZero(mask)
                    count += 1
                
                # Process LAB ranges
                if 'lab' in ranges:
                    mask = cv2.inRange(lab, np.array(ranges['lab'][0]), 
                                    np.array(ranges['lab'][1]))
                    score += cv2.countNonZero(mask)
                    count += 1
                
                # Calculate average score
                if count > 0:
                    color_scores[color_name] = (score / count) / total_pixels
            
            # Get dominant color and confidence
            if color_scores:
                dominant_color = max(color_scores.items(), key=lambda x: x[1])
                confidence = min(1.0, dominant_color[1] * 2)  # Scale confidence
                
                # Debug logging
                logger.debug(f"Color scores: {color_scores}")
                logger.debug(f"Selected color: {dominant_color[0]} with confidence {confidence:.2f}")
                
                return dominant_color[0], confidence
            
            return 'unknown', 0.0
            
        except Exception as e:
            logger.error(f"Error in color detection: {str(e)}")
            return 'unknown', 0.0
    

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for model input"""
        try:
            # Resize image
            resized = cv2.resize(image, (224, 224))
            
            # Apply preprocessing from model info
            mean_values = self.model_info.get('mean_values', [104, 117, 123])
            scale = self.model_info.get('scale_factor', 1.0)
            
            # Create blob with appropriate parameters
            blob = cv2.dnn.blobFromImage(
                resized,
                scale,
                (224, 224),
                mean_values,
                swapRB=True
            )
            
            return blob
            
        except Exception as e:
            logger.error(f"Error preparing image: {str(e)}")
            raise

    def _save_vehicle_image(self, image: np.ndarray) -> Optional[Path]:
        """Save vehicle crop image with enhanced error handling"""
        try:
            # Create timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = self.vehicle_images_path / f"vehicle_{timestamp}.jpg"
            
            # Ensure directory exists
            self.vehicle_images_path.mkdir(parents=True, exist_ok=True)
            
            # Save image with quality parameter
            cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error saving vehicle image: {str(e)}")
            return None

    def _get_vehicle_info(self, class_index: int, confidence: float) -> Dict[str, Any]:
        """Get comprehensive vehicle information from class index"""
        try:
            if self.class_mapping and str(class_index) in self.class_mapping['models']:
                model_info = self.class_mapping['models'][str(class_index)]
                return {
                    'make': model_info['make'],
                    'model': model_info['model'],
                    'type': model_info['type'],
                    'year': model_info['years'][-1] if model_info['years'] else None,
                    'confidence_scores': {
                        'make': float(confidence),
                        'model': float(confidence),
                        'type': float(confidence * 0.9)
                    }
                }
        except Exception as e:
            logger.error(f"Error getting vehicle info for class {class_index}: {str(e)}")
            
        return self._get_fallback_prediction()

    def _get_fallback_prediction(self) -> Dict[str, Any]:
        """Provide fallback predictions when processing fails"""
        return {
            'make': 'Unknown',
            'model': 'Unknown',
            'color': 'unknown',
            'type': 'Unknown',
            'year': None,
            'confidence_scores': {
                'make': 0.0,
                'model': 0.0,
                'color': 0.0,
                'type': 0.0
            },
            'image_path': None
        }