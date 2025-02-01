# app/recognition/basic_recognizer.py

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
from app.recognition import VehicleAttributeRecognizer, VehicleAttributes

logger = logging.getLogger(__name__)

class BasicVehicleRecognizer(VehicleAttributeRecognizer):
    """Basic vehicle attribute recognizer"""
    
    def __init__(self):
        self.confidence_threshold = 0.6
        self.vehicle_images_path = Path("data/vehicle_images")
        self.vehicle_images_path.mkdir(parents=True, exist_ok=True)
        
    def recognize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> VehicleAttributes:
        """Main recognition method with preprocessing"""
        try:
            # Extract vehicle region
            x1, y1, x2, y2 = bbox
            vehicle_crop = image[y1:y2, x1:x2]
            
            # Basic preprocessing
            # 1. Resize if too large
            max_size = 800
            height, width = vehicle_crop.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                vehicle_crop = cv2.resize(vehicle_crop, (new_width, new_height))
            
            # 2. Enhance contrast
            lab = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Save original and enhanced images
            image_path = self._save_vehicle_image(vehicle_crop)
            enhanced_path = None
            if image_path:
                enhanced_path = image_path.parent / f"{image_path.stem}_enhanced.jpg"
                cv2.imwrite(str(enhanced_path), enhanced)
            
            # Detect color from enhanced image
            color, color_confidence = self._detect_color(enhanced)
            
            # Return attributes
            return VehicleAttributes(
                make="Unknown",
                model="Unknown",
                color=color,
                year=None,
                type="Unknown",
                confidence_scores={
                    'color': color_confidence,
                    'make': 0.0,
                    'model': 0.0,
                    'type': 0.0
                },
                image_path=str(image_path) if image_path else None
            )
                
        except Exception as e:
            logger.error(f"Error in basic recognition: {str(e)}")
            return self._get_fallback_result()

    def get_confidence(self) -> float:
        return self.confidence_threshold

    def supports_attribute(self, attribute: str) -> bool:
        return attribute.lower() == 'color'
    
    
    
    
    
    def _detect_color(self, image: np.ndarray) -> Tuple[str, float]:
        """Enhanced color detection with more debugging"""
        try:
            if image is None or image.size == 0:
                return "unknown", 0.0
                
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # More specific yellow ranges
            color_ranges = {
                'yellow': {
                    'hsv': [
                        ([15, 20, 150], [45, 255, 255]),    # Bright yellow
                        ([15, 10, 150], [45, 100, 255]),    # Pale yellow
                        ([15, 0, 150], [45, 50, 255])       # Very pale yellow
                    ],
                    'lab': ([150, 110, 120], [255, 145, 150])
                },
                'silver': {
                    'hsv': [([0, 0, 180], [180, 25, 255])],
                    'lab': ([180, 90, 90], [250, 130, 130])
                },
                'white': {
                    'hsv': [([0, 0, 220], [180, 20, 255])],
                    'lab': ([220, 100, 100], [255, 130, 130])
                },
                'blue': {
                    'hsv': [([100, 80, 50], [130, 255, 255])],
                    'lab': ([0, 115, 0], [150, 140, 110])
                },
                'gray': {
                    'hsv': [([0, 0, 70], [180, 20, 190])],
                    'lab': ([100, 110, 110], [180, 130, 130])
                }
            }
            
            # Calculate weighted color scores
            height, width = image.shape[:2]
            total_pixels = height * width
            min_area_ratio = 0.05  # Reduced minimum area threshold
            
            # Additional preprocessing
            lab_enhanced = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_enhanced)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab_enhanced = cv2.merge((l,a,b))
            
            # Debug: Print mean HSV values
            mean_hsv = cv2.mean(hsv)[:3]
            logger.debug(f"Mean HSV values: H={mean_hsv[0]:.1f}, S={mean_hsv[1]:.1f}, V={mean_hsv[2]:.1f}")
            
            color_scores = {}
            for color_name, ranges in color_ranges.items():
                total_score = 0
                count = 0
                
                if 'hsv' in ranges:
                    hsv_score = 0
                    for range_pair in ranges['hsv']:
                        lower, upper = range_pair
                        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                        score = cv2.countNonZero(mask) / total_pixels
                        hsv_score = max(hsv_score, score)  # Take the highest score from ranges
                    total_score += hsv_score
                    count += 1
                    logger.debug(f"{color_name} HSV score: {hsv_score:.3f}")
                
                if 'lab' in ranges:
                    mask = cv2.inRange(lab, np.array(ranges['lab'][0]), 
                                    np.array(ranges['lab'][1]))
                    lab_score = cv2.countNonZero(mask) / total_pixels
                    total_score += lab_score
                    count += 1
                    logger.debug(f"{color_name} LAB score: {lab_score:.3f}")
                
                if count > 0:
                    avg_score = total_score / count
                    if avg_score > min_area_ratio:
                        color_scores[color_name] = avg_score
                        logger.debug(f"Final {color_name} score: {avg_score:.3f}")
                    else:
                        logger.debug(f"{color_name} score {avg_score:.3f} below threshold {min_area_ratio}")
            
            # Get dominant color
            if color_scores:
                dominant_color = max(color_scores.items(), key=lambda x: x[1])
                confidence = min(1.0, dominant_color[1] * 2)
                
                logger.info(f"Selected color: {dominant_color[0]} with confidence {confidence:.2f}")
                return dominant_color[0], confidence
            else:
                logger.warning("No color scores met the minimum threshold")
                
                # Get the highest score regardless of threshold
                all_scores = {}
                for color_name, ranges in color_ranges.items():
                    total_score = 0
                    count = 0
                    if 'hsv' in ranges:
                        hsv_score = 0
                        for range_pair in ranges['hsv']:
                            lower, upper = range_pair
                            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                            score = cv2.countNonZero(mask) / total_pixels
                            hsv_score = max(hsv_score, score)
                        total_score += hsv_score
                        count += 1
                    if count > 0:
                        all_scores[color_name] = total_score / count
                
                if all_scores:
                    best_color = max(all_scores.items(), key=lambda x: x[1])
                    logger.info(f"Best match (below threshold): {best_color[0]} ({best_color[1]:.3f})")
                    return best_color[0], best_color[1]
            
            return "unknown", 0.0
            
        except Exception as e:
            logger.error(f"Error in color detection: {str(e)}")
            return "unknown", 0.0
    
    
    

    def _save_vehicle_image(self, image: np.ndarray) -> Optional[Path]:
        """Save vehicle crop image"""
        try:
            if image is None or image.size == 0:
                logger.warning("Invalid image provided for saving")
                return None
                
            # Create output directory if needed
            self.vehicle_images_path.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = self.vehicle_images_path / f"vehicle_{timestamp}.jpg"
            
            # Save image
            success = cv2.imwrite(str(image_path), image)
            if not success:
                logger.error("Failed to save image")
                return None
                
            logger.debug(f"Saved vehicle image to {image_path}")
            return image_path
            
        except Exception as e:
            logger.error(f"Error saving vehicle image: {str(e)}")
            return None
    
    def _get_fallback_result(self) -> VehicleAttributes:
        """Return fallback result when processing fails"""
        return VehicleAttributes(
            make="Unknown",
            model="Unknown",
            color="unknown",
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