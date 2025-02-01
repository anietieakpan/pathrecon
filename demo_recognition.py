# demo_recognition.py

import cv2
import logging
from pathlib import Path
from app.recognition import VehicleRecognizerFactory, RecognitionType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_recognition_types():
    """Demo different recognition types"""
    # Test image path
    image_path = "data/examples/oneline_images/example1.jpeg"
    if not Path(image_path).exists():
        logger.error(f"Test image not found: {image_path}")
        return
    
    # Load test image
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Failed to load image")
        return
    
    # Create test bbox
    height, width = image.shape[:2]
    bbox = (0, 0, width, height)
    
    # Test each recognition type
    for rec_type in RecognitionType:
        try:
            logger.info(f"\nTesting {rec_type.value} recognition...")
            
            # Create recognizer
            recognizer = VehicleRecognizerFactory.create_recognizer(rec_type)
            
            # Get attributes
            attributes = recognizer.recognize(image, bbox)
            
            # Print results
            logger.info("Results:")
            logger.info(f"Color: {attributes.color} ({attributes.confidence_scores['color']:.2f})")
            logger.info(f"Make: {attributes.make} ({attributes.confidence_scores['make']:.2f})")
            logger.info(f"Model: {attributes.model} ({attributes.confidence_scores['model']:.2f})")
            logger.info(f"Type: {attributes.type} ({attributes.confidence_scores['type']:.2f})")
            if attributes.year:
                logger.info(f"Year: {attributes.year}")
            
            # Print supported attributes
            supported = [attr for attr in ['color', 'make', 'model', 'type', 'year'] 
                        if recognizer.supports_attribute(attr)]
            logger.info(f"Supported attributes: {supported}")
            
        except Exception as e:
            logger.error(f"Error testing {rec_type.value}: {str(e)}")

if __name__ == "__main__":
    demo_recognition_types()