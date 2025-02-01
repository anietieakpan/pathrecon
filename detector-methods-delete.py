# app/detection/detector.py

import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
import warnings
import os
import base64
import traceback
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from .vehicle_classifier import VehicleClassifier

logger = logging.getLogger(__name__)

class LicensePlateDetector:
    def __init__(self, database_factory=None):
        """Initialize detector with optional database factory"""
        self.detector = pipeline("number_plate_detection_and_reading", image_loader="opencv")
        self.vehicle_classifier = VehicleClassifier()
        self.database_factory = database_factory
        
        if self.database_factory:
            self.initialize_databases()
        
        self.is_processing = False
        self.detected_plates = []

    def initialize_databases(self):
        """Initialize database connections"""
        if self.database_factory:
            self.databases = self.database_factory.get_all_databases()
        else:
            self.databases = None

    def process_frame(self, frame, frame_size=None) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame to detect license plates and vehicle details"""
        try:
            if frame_size:
                frame = cv2.resize(frame, frame_size)

            # Detect license plates
            results = self.detector([frame])
            (images, bboxs, points, zones,
             region_ids, region_names,
             count_lines, confidences, texts) = unzip(results)

            visualization = frame.copy()
            detections = []

            if bboxs and len(bboxs[0]) > 0:
                for i, bbox in enumerate(bboxs[0]):
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    det_confidence = float(bbox[4])

                    # Process vehicle image around license plate
                    vehicle_bbox = self._expand_bbox(frame, (x1, y1, x2, y2))
                    vehicle_details = self.vehicle_classifier.process_vehicle_image(
                        frame, vehicle_bbox
                    )

                    if texts and len(texts[0]) > i:
                        text = texts[0][i]
                        if isinstance(text, list):
                            text = ' '.join(text)
                        
                        detection_info = {
                            'text': text,
                            'confidence': det_confidence,
                            'bbox': (x1, y1, x2, y2),
                            'timestamp_utc': datetime.utcnow(),
                            'timestamp_local': datetime.now(),
                            'vehicle_details': vehicle_details
                        }
                        detections.append(detection_info)

                        # Draw detections on visualization
                        self._draw_detection(
                            visualization, 
                            detection_info,
                            (x1, y1, x2, y2)
                        )

                        # Store detection in databases
                        if self.databases:
                            self._store_detection(detection_info)

            return visualization, detections

        except Exception as e:
            logger.error(f"Error in process_frame: {str(e)}")
            traceback.print_exc()
            return frame.copy(), []

    def _expand_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Expand bounding box to include more of the vehicle"""
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        
        # Expand box (adjust these values based on your needs)
        expand_x = int((x2 - x1) * 1.5)  # 150% wider
        expand_y = int((y2 - y1) * 2.0)  # 200% taller
        
        x1 = max(0, x1 - expand_x)
        x2 = min(width, x2 + expand_x)
        y1 = max(0, y1 - expand_y)
        y2 = min(height, y2 + expand_y)
        
        return (x1, y1, x2, y2)

    def _draw_detection(self, image: np.ndarray, detection: Dict, bbox: Tuple[int, int, int, int]):
        """Draw detection information on image"""
        x1, y1, x2, y2 = bbox
        
        # Draw license plate box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw text with details
        vehicle_details = detection.get('vehicle_details', {})
        text_lines = [
            f"Plate: {detection['text']} ({detection['confidence']:.2f})",
            f"Make: {vehicle_details.get('make', 'Unknown')}",
            f"Model: {vehicle_details.get('model', 'Unknown')}",
            f"Color: {vehicle_details.get('color', 'Unknown')}",
            f"Type: {vehicle_details.get('type', 'Unknown')}"
        ]
        
        y_offset = y1 - 10
        for line in text_lines:
            y_offset -= 20
            cv2.putText(
                image,
                line,
                (x1, max(20, y_offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    def _store_detection(self, detection: Dict):
        """Store detection in databases"""
        try:
            if 'timeseries' in self.databases:
                self.databases['timeseries'].insert_detection(detection)
                
            if 'postgres' in self.databases:
                self.databases['postgres'].insert_detection(detection)
                
        except Exception as e:
            logger.error(f"Error storing detection: {str(e)}")

    # ... (keep existing methods for video/camera handling)
    
    
    
    
    
    
    
    
    # ---------------another one - aniix!!!
    
    
    def process_frame(self, frame, frame_size=None):
        try:
            if frame_size:
                frame = cv2.resize(frame, frame_size)
            else:
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))

            temp_frame_path = str(self.temp_dir / "temp_frame.jpg")
            cv2.imwrite(temp_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            results = self.detector([temp_frame_path])
            (images, bboxs, points, zones,
            region_ids, region_names,
            count_lines, confidences, texts) = unzip(results)

            visualization = frame.copy()
            detections = []

            if bboxs and len(bboxs[0]) > 0:
                for i, bbox in enumerate(bboxs[0][:self.max_detections_per_frame]):
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    det_confidence = float(bbox[4])

                    if det_confidence < self.confidence_threshold:
                        continue

                    # Get vehicle details
                    vehicle_bbox = self._expand_bbox(frame, (x1, y1, x2, y2))
                    vehicle_details = self.vehicle_classifier.process_vehicle_image(
                        frame, vehicle_bbox
                    )

                    if texts and len(texts[0]) > i:
                        text = texts[0][i]
                        if isinstance(text, list):
                            text = ' '.join(text)

                        detection_info = {
                            'text': text,
                            'confidence': det_confidence,
                            'bbox': (x1, y1, x2, y2)
                        }
                        detections.append(detection_info)

                        # Draw detection visualization
                        self._draw_detection(visualization, detection_info, vehicle_details, (x1, y1, x2, y2))

                        # Store detection in databases
                        if self.databases:
                            self._store_detection(detection_info, vehicle_details)

            return visualization, detections

        except Exception as e:
            logging.error(f"Error in process_frame: {str(e)}")
            return frame.copy(), []

    def _expand_bbox(self, image, bbox):
        """Expand bounding box to include more of the vehicle"""
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        
        # Expand box by 50% in each direction
        expand_x = int((x2 - x1) * 0.5)
        expand_y = int((y2 - y1) * 0.5)
        
        new_x1 = max(0, x1 - expand_x)
        new_x2 = min(width, x2 + expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_y2 = min(height, y2 + expand_y)
        
        return (new_x1, new_y1, new_x2, new_y2)

    def _draw_detection(self, image, detection, vehicle_details, bbox):
        """Draw detection and vehicle information on image"""
        x1, y1, x2, y2 = bbox
        
        # Draw license plate box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Prepare text lines
        text_lines = [
            f"{detection['text']} ({detection['confidence']:.2f})"
        ]
        
        if vehicle_details:
            text_lines.extend([
                f"Make: {vehicle_details.get('make', 'Unknown')}",
                f"Model: {vehicle_details.get('model', 'Unknown')}",
                f"Color: {vehicle_details.get('color', 'Unknown')}",
                f"Type: {vehicle_details.get('type', 'Unknown')}"
            ])

        # Draw text background
        text_size = cv2.getTextSize(text_lines[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(image,
                    (x1, y1 - text_size[1] * (len(text_lines) + 1)),
                    (x1 + text_size[0], y1),
                    (0, 255, 0),
                    -1)
        
        # Draw text lines
        y_pos = y1 - 10
        for line in reversed(text_lines):
            cv2.putText(image,
                      line,
                      (x1, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      1.0,
                      (0, 0, 0),
                      2)
            y_pos -= text_size[1] + 5