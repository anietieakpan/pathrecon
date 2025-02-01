# test_vehicle_detection.py

import cv2
import logging
from pathlib import Path
import numpy as np
from app.detection.vehicle_detector import VehicleObjectDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_detection_image(frame, detections, output_path):
    """Save frame with detections to file"""
    frame_with_detections = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_name = det['class']
        
        # Draw bounding box
        cv2.rectangle(frame_with_detections, 
                    (x1, y1), (x2, y2),
                    (255, 0, 0),  # Blue color
                    2)
        
        # Add label
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame_with_detections,
                  label,
                  (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  (255, 0, 0),
                  2)
    
    cv2.imwrite(output_path, frame_with_detections)
    logger.info(f"Saved detection image to: {output_path}")

def test_vehicle_detection():
    """Test vehicle detection on video/camera feed"""
    # Initialize detector
    detector = VehicleObjectDetector(confidence_threshold=0.3)
    logger.info("Vehicle detector initialized")
    
    # Create output directory
    output_dir = Path("output/vehicle_detection_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to open video file first, if not found use camera
    video_path = "uploads/deneme.mp4"  # Adjust path as needed
    if Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        logger.info(f"Opened video file: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        logger.info("Using camera feed")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 != 0:  # Process every 10th frame
                continue
            
            # Detect vehicles
            detections = detector.detect_vehicles(frame)
            logger.info(f"Frame {frame_count}: Found {len(detections)} vehicles")
            
            # Log detection details
            for i, det in enumerate(detections):
                logger.info(f"Detection {i + 1}:")
                logger.info(f"  Class: {det['class']}")
                logger.info(f"  Confidence: {det['confidence']:.2f}")
                logger.info(f"  Bounding Box: {det['bbox']}")
            
            # Save frame with detections
            if len(detections) > 0:
                output_path = output_dir / f"detection_frame_{frame_count}.jpg"
                save_detection_image(frame, detections, output_path)
            
            # Stop after processing 100 frames
            if frame_count >= 1000:
                break
                
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        raise
    
    finally:
        cap.release()
        logger.info("Test completed")

def test_single_image():
    """Test vehicle detection on a single image"""
    # Initialize detector
    detector = VehicleObjectDetector(confidence_threshold=0.3)
    logger.info("Vehicle detector initialized")
    
    # Create output directory
    output_dir = Path("output/vehicle_detection_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test image
    image_paths = [
        "data/examples/oneline_images/example1.jpeg",
        "data/examples/oneline_images/example2.jpeg",
        # Add more test image paths here
    ]
    
    for image_path in image_paths:
        if not Path(image_path).exists():
            logger.error(f"Test image not found: {image_path}")
            continue
        
        # Read and process image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            continue
            
        try:
            # Detect vehicles
            detections = detector.detect_vehicles(image)
            logger.info(f"Image {image_path}: Found {len(detections)} vehicles")
            
            # Log detection details
            for i, det in enumerate(detections):
                logger.info(f"Detection {i + 1}:")
                logger.info(f"  Class: {det['class']}")
                logger.info(f"  Confidence: {det['confidence']:.2f}")
                logger.info(f"  Bounding Box: {det['bbox']}")
            
            # Save results
            if len(detections) > 0:
                output_path = output_dir / f"detection_{Path(image_path).stem}.jpg"
                save_detection_image(image, detections, output_path)
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--image':
        test_single_image()
    else:
        test_vehicle_detection()