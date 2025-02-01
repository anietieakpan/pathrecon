# example_usage.py

import cv2
import logging
from app.recognition import PreTrainedVehicleRecognizer
from datetime import datetime
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_vehicle_detection(image_path, license_plate_bbox):
    """
    Process vehicle detection and recognize attributes
    
    Args:
        image_path: Path to the image file
        license_plate_bbox: Tuple of (x1, y1, x2, y2) for license plate location
    """
    try:
        # Initialize recognizer
        recognizer = PreTrainedVehicleRecognizer()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Expand bbox to include whole vehicle
        # Usually the vehicle is above and around the license plate
        x1, y1, x2, y2 = license_plate_bbox
        img_height, img_width = image.shape[:2]
        
        # Expand bbox (adjust these values based on your needs)
        vehicle_x1 = max(0, x1 - int((x2 - x1) * 1.5))
        vehicle_y1 = max(0, y1 - int((y2 - y1) * 4))  # More expansion upward
        vehicle_x2 = min(img_width, x2 + int((x2 - x1) * 1.5))
        vehicle_y2 = min(img_height, y2 + int((y2 - y1) * 0.5))
        
        vehicle_bbox = (vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2)
        
        # Recognize vehicle attributes
        attributes = recognizer.recognize(image, vehicle_bbox)
        
        # Create detection data
        detection_data = {
            'text': 'ABC123',  # Replace with actual plate text
            'confidence': 0.95,  # Replace with actual confidence
            'timestamp_utc': datetime.now(pytz.UTC),
            'timestamp_local': datetime.now(pytz.UTC).astimezone(pytz.timezone('Africa/Johannesburg')),
            'vehicle_details': {
                'make': attributes.make,
                'model': attributes.model,
                'color': attributes.color,
                'year': attributes.year,
                'type': attributes.type,
                'image_path': attributes.image_path,
                'confidence_scores': attributes.confidence_scores
            }
        }
        
        # Log results
        logger.info("Vehicle Recognition Results:")
        logger.info(f"Make: {attributes.make} ({attributes.confidence_scores['make']:.2f})")
        logger.info(f"Model: {attributes.model} ({attributes.confidence_scores['model']:.2f})")
        logger.info(f"Color: {attributes.color} ({attributes.confidence_scores['color']:.2f})")
        logger.info(f"Type: {attributes.type} ({attributes.confidence_scores['type']:.2f})")
        if attributes.year:
            logger.info(f"Year: {attributes.year}")
        
        return detection_data
        
    except Exception as e:
        logger.error(f"Error processing vehicle detection: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Example image path and license plate bbox
    image_path = "test.jpg"
    license_plate_bbox = (100, 100, 200, 150)  # Example bbox
    
    result = process_vehicle_detection(image_path, license_plate_bbox)
    
    if result:
        from pprint import pprint
        print("\nDetection Data:")
        pprint(result)