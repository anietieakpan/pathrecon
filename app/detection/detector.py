# app/detection/detector.py

import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
from pathlib import Path
import tempfile
import warnings
import os
import base64
import traceback
from picamera2 import Picamera2
from flask import current_app
import time
import logging
import pytz
from datetime import datetime
from .vehicle_classifier import VehicleClassifier
from typing import Dict, List, Tuple, Optional, Any

from app.database.factory import DatabaseFactory

from .vehicle_detector import VehicleObjectDetector

from app.recognition import VehicleRecognizerFactory, RecognitionType



logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning, 
                       message='Implicit dimension choice for softmax.*')
warnings.filterwarnings('ignore', category=UserWarning, 
                       message='Creating a tensor from a list of numpy.ndarrays is extremely slow.*')

class LicensePlateDetector:
    def __init__(self, database_factory):
        logging.info("Initializing LicensePlateDetector")
        # Path('output').mkdir(exist_ok=True)
        
        #  Initialize directories
        Path('output').mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        
        
        # Initialize license plate detector
        logger.info("Loading license plate detector...")
        self.detector = pipeline("number_plate_detection_and_reading", image_loader="opencv")
        
        
        # Initialize vehicle detector
        self.vehicle_detector = VehicleObjectDetector(confidence_threshold=0.3)
        
        # Initialize vehicle classifier
        logger.info("Loading vehicle classifier...")
        self.vehicle_classifier = VehicleClassifier()
        
        
         # Database initialization
        self.database_factory = database_factory
        self.databases = None
        
        
        # Video capture attributes
        self.cap = None
        self.current_frame = None
        self.is_processing = False
        self.detected_plates = []

        # Processing parameters
        self.frame_count = 0
        self.last_process_time = 0
        self.frame_skip = 2
        self.resize_width = 640
        self.resize_height = 480
        self.confidence_threshold = 0.5
        self.max_detections_per_frame = 5
        self.process_every_n_seconds = 1
        
        
        # Time zone setup
        self.local_tz = pytz.timezone('Africa/Johannesburg')
        
        
         # Create vehicle images directory
        self.vehicle_images_path = Path("data/vehicle_images")
        self.vehicle_images_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases if factory provided
        if self.database_factory:
            self.initialize_databases()
            
        # Initialize vehicle recognizer (using basic by default)
        self.vehicle_recognizer = VehicleRecognizerFactory.create_recognizer(
            RecognitionType.PRETRAINED
        )
        
        logger.info(f"Using {self.vehicle_recognizer.__class__.__name__} for vehicle recognition")
            
            
        logger.info("LicensePlateDetector initialization complete")
        

    def initialize_databases(self):
        """Initialize database connections"""
        try:
            logging.info("Attempting to get all databases")
            self.databases = self.database_factory.get_all_databases()
            logging.info(f"Databases retrieved: {list(self.databases.keys())}")
            
            for db_name, db in self.databases.items():
                try:
                    logging.info(f"Attempting to connect to {db_name} database")
                    db.connect()
                    logging.info(f"Successfully connected to {db_name} database")
                    print(f"Successfully connected to {db_name} database")
                except Exception as db_error:
                    logging.error(f"Error connecting to {db_name} database: {str(db_error)}")
        except Exception as e:
            logging.error(f"Error initializing databases: {str(e)}", exc_info=True)

    def __del__(self):
        """Cleanup database connections"""
        if hasattr(self, 'databases') and self.databases:
            for db in self.databases.values():
                db.disconnect()
                
                
 

    def _prepare_detection_data(self, detection, vehicle_details):
        """Prepare detection data for database storage"""
        utc_time = datetime.now(pytz.UTC)
        local_time = utc_time.astimezone(self.local_tz)
        
        # Base detection data
        base_data = {
            'text': detection['text'],
            'confidence': detection['confidence'],
            'timestamp_utc': utc_time,
            'timestamp_local': local_time,
        }
        
        # Add vehicle details if available
        if vehicle_details:
            base_data.update({
                'vehicle_make': vehicle_details.get('make'),
                'vehicle_model': vehicle_details.get('model'),
                'vehicle_color': vehicle_details.get('color'),
                'vehicle_type': vehicle_details.get('type'),
                'vehicle_year': vehicle_details.get('year'),
                'vehicle_image_path': vehicle_details.get('image_path'),
                'vehicle_confidence_scores': vehicle_details.get('confidence_scores')
            })
        
        # Create InfluxDB-specific version (with ISO format timestamps)
        influx_data = base_data.copy()
        influx_data['timestamp_utc'] = utc_time.isoformat()
        influx_data['timestamp_local'] = local_time.isoformat()
        
        return base_data, influx_data
    
    
        
        # Add vehicle details if available
        if vehicle_details:
            base_data.update({
                'vehicle_make': vehicle_details.get('make'),
                'vehicle_model': vehicle_details.get('model'),
                'vehicle_color': vehicle_details.get('color'),
                'vehicle_type': vehicle_details.get('type'),
                'vehicle_year': vehicle_details.get('year'),
                'vehicle_image_path': vehicle_details.get('image_path'),
                'vehicle_confidence_scores': vehicle_details.get('confidence_scores')
            })
        
        # Create InfluxDB-specific version (with ISO format timestamps)
        influx_data = base_data.copy()
        influx_data['timestamp_utc'] = utc_time.isoformat()
        influx_data['timestamp_local'] = local_time.isoformat()
        
        return base_data, influx_data
                
    
    def process_frame(self, frame, frame_size=None):
        """Process frame for both vehicles and license plates"""
        try:
            if frame is None:
                logger.error("Received empty frame")
                return None, []

            if frame_size:
                frame = cv2.resize(frame, frame_size)

            visualization = frame.copy()
            all_detections = []

            # Vehicle Detection
            logger.info("Running vehicle detection...")
            vehicle_detections = self.vehicle_detector.detect_vehicles(frame)
            logger.info(f"Found {len(vehicle_detections)} vehicles")

            # Store vehicle regions for later use
            vehicle_regions = []
            
            # Process and draw vehicle detections
            for veh in vehicle_detections:
                vx1, vy1, vx2, vy2 = veh['bbox']
                veh_class = veh['class']
                veh_conf = veh['confidence']
                
                # Draw blue box for vehicle
                cv2.rectangle(visualization, 
                            (vx1, vy1), (vx2, vy2),
                            (255, 0, 0),  # Blue
                            3)  # Thicker line
                
                # Draw vehicle label
                veh_label = f"{veh_class} ({veh_conf:.2f})"
                label_size = cv2.getTextSize(veh_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Background for vehicle label
                cv2.rectangle(visualization,
                            (vx1, vy1 - label_size[1] - 10),
                            (vx1 + label_size[0], vy1),
                            (255, 0, 0),
                            -1)
                
                # Vehicle label text
                cv2.putText(visualization,
                        veh_label,
                        (vx1, vy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),  # White text
                        2)
                
                # Store vehicle region
                vehicle_regions.append({
                    'bbox': (vx1, vy1, vx2, vy2),
                    'class': veh_class,
                    'confidence': veh_conf,
                    'image': frame[vy1:vy2, vx1:vx2]
                })

            # License Plate Detection
            temp_frame_path = str(self.temp_dir / "temp_frame.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            results = self.detector([temp_frame_path])
            images, bboxs, points, zones, region_ids, region_names, count_lines, confidences, texts = unzip(results)

            if bboxs and len(bboxs[0]) > 0:
                for i, bbox in enumerate(bboxs[0]):
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    plate_conf = float(bbox[4])
                    
                    # Get plate text
                    plate_text = ''
                    if texts and len(texts[0]) > i:
                        plate_text = texts[0][i]
                        if isinstance(plate_text, list):
                            plate_text = ' '.join(text)

                    # Find associated vehicle
                    associated_vehicle = None
                    for veh in vehicle_regions:
                        vx1, vy1, vx2, vy2 = veh['bbox']
                        # Check if plate is within vehicle bounds (with some margin)
                        margin = 20
                        if (x1 >= vx1-margin and x2 <= vx2+margin and 
                            y1 >= vy1-margin and y2 <= vy2+margin):
                            associated_vehicle = veh
                            break

                    # Get vehicle details if we have an associated vehicle
                    vehicle_details = None
                    if associated_vehicle:
                        vehicle_crop = associated_vehicle['image']
                        if vehicle_crop is not None and vehicle_crop.size > 0:
                            vehicle_details = self._get_vehicle_details(vehicle_crop, (x1-vx1, y1-vy1, x2-vx1, y2-vy1))

                    # Draw green box for license plate
                    cv2.rectangle(visualization, 
                                (x1, y1), (x2, y2),
                                (0, 255, 0),  # Green
                                2)

                    # Create detection info
                    detection_info = {
                        'text': plate_text,
                        'confidence': plate_conf,
                        'bbox': (x1, y1, x2, y2),
                        'vehicle_type': associated_vehicle['class'] if associated_vehicle else 'unknown',
                        'vehicle_confidence': associated_vehicle['confidence'] if associated_vehicle else 0.0,
                        'vehicle_details': vehicle_details
                    }

                    # Draw plate info
                    y_offset = y1 - 10
                    plate_label = f"Plate: {plate_text} ({plate_conf:.2f})"
                    cv2.putText(visualization,
                            plate_label,
                            (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2)

                    # Draw vehicle details if available
                    if vehicle_details:
                        details_text = f"{vehicle_details['color']} {vehicle_details['make']} {vehicle_details['model']}"
                        y_offset -= 20
                        cv2.putText(visualization,
                                details_text,
                                (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),  # Red
                                2)

                    all_detections.append(detection_info)

                    # Store if new
                    if self.databases:
                        self._store_detection(detection_info, vehicle_details)

            

            return visualization, all_detections

        except Exception as e:
            logger.error(f"Error in process_frame: {str(e)}")
            traceback.print_exc()
            return frame.copy() if frame is not None else None, []
    
    
    

    def get_frame(self):
        """Get processed frame with error handling"""
        try:
            if not self.cap or not self.is_processing:
                return None
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.is_processing = False
                return None
        
            processed_frame, detections = self.process_frame(frame)
            if processed_frame is None:
                return None
            
            ret, jpeg = cv2.imencode('.jpg', processed_frame)
            if not ret:
                return None
            
            return jpeg.tobytes()
        
        except Exception as e:
            logger.error(f"Error in get_frame: {str(e)}")
            return None
    

        
    def process_image(self, image):
        """Process a single image for both vehicles and license plates"""
        try:
            logger.info("Starting image processing...")
            
            # Step 1: Detect Vehicles
            logger.info("Running vehicle detection...")
            vehicle_detections = self.vehicle_detector.detect_vehicles(image)
            logger.info(f"Found {len(vehicle_detections)} vehicles")
            
            visualization = image.copy()
            all_detections = []
            vehicle_regions = []
            
            # Process vehicle detections
            for veh in vehicle_detections:
                vx1, vy1, vx2, vy2 = veh['bbox']
                veh_class = veh['class']
                veh_conf = veh['confidence']
                
                # Draw blue box for vehicle
                cv2.rectangle(visualization, 
                            (vx1, vy1), (vx2, vy2),
                            (255, 0, 0),  # Blue
                            3)  # Thicker line
                
                # Draw vehicle label
                veh_label = f"{veh_class} ({veh_conf:.2f})"
                label_size = cv2.getTextSize(veh_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Background for vehicle label
                cv2.rectangle(visualization,
                            (vx1, vy1 - label_size[1] - 10),
                            (vx1 + label_size[0], vy1),
                            (255, 0, 0),
                            -1)
                
                # Vehicle label text
                cv2.putText(visualization,
                        veh_label,
                        (vx1, vy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),  # White text
                        2)
                
                # Store vehicle region
                vehicle_regions.append({
                    'bbox': (vx1, vy1, vx2, vy2),
                    'class': veh_class,
                    'confidence': veh_conf,
                    'image': image[vy1:vy2, vx1:vx2]
                })
            
            # Step 2: License Plate Detection
            temp_image_path = str(self.temp_dir / "temp_image.jpg")
            cv2.imwrite(temp_image_path, image)

            results = self.detector([temp_image_path])
            images, bboxs, points, zones, region_ids, region_names, count_lines, confidences, texts = unzip(results)

            if bboxs and len(bboxs[0]) > 0:
                for i, bbox in enumerate(bboxs[0]):
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    plate_conf = float(bbox[4])
                    
                    # Get plate text
                    plate_text = ''
                    if texts and len(texts[0]) > i:
                        plate_text = texts[0][i]
                        if isinstance(plate_text, list):
                            plate_text = ' '.join(plate_text)

                    # Find associated vehicle
                    associated_vehicle = None
                    for veh in vehicle_regions:
                        vx1, vy1, vx2, vy2 = veh['bbox']
                        # Check if plate is within vehicle bounds (with some margin)
                        margin = 50  # Increased margin for better association
                        if (x1 >= vx1-margin and x2 <= vx2+margin and 
                            y1 >= vy1-margin and y2 <= vy2+margin):
                            associated_vehicle = veh
                            break

                    # Get vehicle details if we have an associated vehicle
                    vehicle_details = None
                    if associated_vehicle:
                        vehicle_crop = associated_vehicle['image']
                        if vehicle_crop is not None and vehicle_crop.size > 0:
                            vehicle_details = self._get_vehicle_details(vehicle_crop, (x1-vx1, y1-vy1, x2-vx1, y2-vy1))

                    # Draw green box for license plate
                    cv2.rectangle(visualization, 
                                (x1, y1), (x2, y2),
                                (0, 255, 0),  # Green
                                2)

                    # Create detection info
                    detection_info = {
                        'text': plate_text,
                        'confidence': plate_conf,
                        'bbox': (x1, y1, x2, y2),
                        'vehicle_type': associated_vehicle['class'] if associated_vehicle else 'unknown',
                        'vehicle_confidence': associated_vehicle['confidence'] if associated_vehicle else 0.0,
                        'vehicle_details': vehicle_details
                    }

                    # Draw plate info
                    y_offset = y1 - 10
                    plate_label = f"Plate: {plate_text} ({plate_conf:.2f})"
                    cv2.putText(visualization,
                            plate_label,
                            (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2)

                    # Draw vehicle details if available
                    if vehicle_details:
                        details_text = f"{vehicle_details['color']} {vehicle_details['make']} {vehicle_details['model']}"
                        y_offset -= 20
                        cv2.putText(visualization,
                                details_text,
                                (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),  # Red
                                2)

                    all_detections.append(detection_info)

            # Encode the result
            _, buffer = cv2.imencode('.jpg', visualization)
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            return encoded_image, all_detections

        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            traceback.print_exc()
            return None, []
    
    
           
    
    def _expand_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
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

    def _draw_detection(self, image: np.ndarray, detection: Dict):
        """Draw detection information on image"""
        x1, y1, x2, y2 = detection['bbox']
        
        # Draw license plate box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Prepare text lines
        text_lines = [
            f"Plate: {detection['text']} ({detection['confidence']:.2f})"
        ]
        
        if 'vehicle_details' in detection:
            vd = detection['vehicle_details']
            text_lines.extend([
                f"Make: {vd.get('make', 'Unknown')}",
                f"Model: {vd.get('model', 'Unknown')}",
                f"Color: {vd.get('color', 'Unknown')}",
                f"Type: {vd.get('type', 'Unknown')}"
            ])
        
        # Draw text with background
        y_offset = y1 - 10
        for line in reversed(text_lines):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background rectangle
            cv2.rectangle(image,
                        (x1, y_offset - text_size[1] - 5),
                        (x1 + text_size[0], y_offset + 5),
                        (0, 255, 0),
                        -1)
            
            # Text
            cv2.putText(image,
                      line,
                      (x1, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7,
                      (0, 0, 0),
                      2)
            
            y_offset -= text_size[1] + 10

    def _store_detection(self, detection, vehicle_details=None):
        """Store detection in all configured databases"""
        if not self.databases:
            return

        try:
            # Prepare data for different databases
            postgres_data, influx_data = self._prepare_detection_data(detection, vehicle_details)

            # Store in InfluxDB
            if 'timeseries' in self.databases:
                try:
                    self.databases['timeseries'].insert_detection(influx_data)
                    logging.info("Successfully inserted detection into timeseries database")
                except Exception as e:
                    logging.error(f"Error storing in InfluxDB: {str(e)}")

            # Store in PostgreSQL
            if 'postgres' in self.databases:
                try:
                    self.databases['postgres'].insert_detection(postgres_data)
                    logging.info("Successfully inserted detection into postgres database")
                except Exception as e:
                    logging.error(f"Error storing in PostgreSQL: {str(e)}")

        except Exception as e:
            logging.error(f"Error in _store_detection: {str(e)}") 
            
            
    def start_video_capture(self, video_path):
        """Start video capture from file"""
        self.detected_plates = []

        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Started video capture. FPS: {self.fps}, Resolution: {self.frame_width}x{self.frame_height}")
        self.is_processing = True
        return self

    
    
    def stop_video_capture(self):
        if self.cap:
            self.cap.release()
        self.is_processing = False

    def get_detected_plates(self):
        return self.detected_plates
    
    def start_camera_capture(self):
        if hasattr(self, 'picam2') and self.picam2 is not None:
            print("Camera already initialized, stopping previous instance")
            self.stop_camera_capture()
    
        try:
            self.detected_plates = []
            self.picam2 = Picamera2()
            print("Picamera2 instance created")
            self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
            print("Camera configured")
            self.picam2.start()
            print("Camera started")
            self.is_processing = True
            print("Camera capture started successfully")
            return self
        except Exception as e:
            print(f"Error in start_camera_capture: {str(e)}")
            self.picam2 = None
            raise
        
    def get_camera_frame(self):
        if not self.is_processing:
            return None

        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        self.frame_count += 1
        current_time = time.time()

        processed_frame = frame
        detections = []

        if (self.frame_count % self.frame_skip == 0 and
            current_time - self.last_process_time >= self.process_every_n_seconds):
            self.last_process_time = current_time
            processed_frame, detections = self.process_frame(frame)

        if detections:
            for det in detections:
                if not any(existing['text'] == det['text'] for existing in self.detected_plates):
                    self.detected_plates.append(det)

        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        return jpeg.tobytes()

    def stop_camera_capture(self):
        if hasattr(self, 'picam2'):
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                print(f"Error stopping camera: {str(e)}")
        self.is_processing = False
        self.picam2 = None  # Ensure the picam2 attribute is cleared
        
        
        
        
    def _get_vehicle_details(self, vehicle_crop: np.ndarray, plate_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Get vehicle details using the pre-trained recognizer"""
        try:
            if vehicle_crop is None or vehicle_crop.size == 0:
                logger.warning("Invalid vehicle crop provided")
                return None

            # Convert relative plate bbox to vehicle crop coordinates
            x1, y1, x2, y2 = plate_bbox
            vehicle_height, vehicle_width = vehicle_crop.shape[:2]
            
            # Expanded bbox for better vehicle recognition
            expanded_bbox = self._expand_bbox(
                vehicle_crop,
                (x1, y1, x2, y2)
            )
            
            # Get attributes using pre-trained recognizer
            attributes = self.vehicle_recognizer.recognize(vehicle_crop, expanded_bbox)
            
            return {
                'make': attributes.make,
                'model': attributes.model,
                'color': attributes.color,
                'year': attributes.year,
                'type': attributes.type,
                'image_path': attributes.image_path,
                'confidence_scores': attributes.confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Error getting vehicle details: {str(e)}")
            return None
        
        
    def update_config(self, config):
        if 'FRAME_SKIP' in config:
            self.frame_skip = config['FRAME_SKIP']
        if 'RESIZE_WIDTH' in config:
            self.resize_width = config['RESIZE_WIDTH']
        if 'RESIZE_HEIGHT' in config:
            self.resize_height = config['RESIZE_HEIGHT']
        if 'CONFIDENCE_THRESHOLD' in config:
            self.confidence_threshold = config['CONFIDENCE_THRESHOLD']
        if 'MAX_DETECTIONS_PER_FRAME' in config:
            self.max_detections_per_frame = config['MAX_DETECTIONS_PER_FRAME']
        if 'PROCESS_EVERY_N_SECONDS' in config:
            self.process_every_n_seconds = config['PROCESS_EVERY_N_SECONDS']

            
    
    