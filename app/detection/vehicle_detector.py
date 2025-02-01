# app/detection/vehicle_detector.py

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import torch

logger = logging.getLogger(__name__)


class VehicleObjectDetector:
    """Handles vehicle object detection using YOLOv8"""
    
    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.initialized = False
        
        # COCO class indices for vehicles
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model
            self.initialized = True
            logger.info(f"Vehicle detector initialized successfully using {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing vehicle detector: {str(e)}")
            self.model = None

    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """Detect vehicles in an image"""
        if not self.initialized or self.model is None:
            logger.warning("Vehicle detector not initialized, skipping detection")
            return []
            
        try:
            # Make predictions
            results = self.model(image, verbose=False)  # Disable progress bar
            
            # Process detections
            detections = []
            
            # Get the first result (only processing single image)
            result = results[0]
            
            for box in result.boxes:
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Only process if it's a vehicle class and meets confidence threshold
                if cls in self.vehicle_classes and conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'class': self.vehicle_classes[cls],
                        'image': image[y1:y2, x1:x2].copy()  # Get vehicle crop
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting vehicles: {str(e)}")
            return []

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels for detected vehicles"""
        draw_img = image.copy()
        
        try:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class']} {det['confidence']:.2f}"
                
                # Draw rectangle
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Calculate label dimensions
                font_scale = 0.6
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw label background
                cv2.rectangle(
                    draw_img,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    (255, 0, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    draw_img,
                    label,
                    (x1, y1 - baseline - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
                
            return draw_img
            
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return image