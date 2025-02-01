# test_recognition.py

import cv2
import logging
import numpy as np
from pathlib import Path
from app.recognition import VehicleRecognizerFactory, RecognitionType

# Set up logging to show debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_vehicle_recognition():
    """Test vehicle recognition with enhanced debugging"""
    # Initialize recognizer
    recognizer = VehicleRecognizerFactory.create_recognizer(RecognitionType.BASIC)
    logger.info(f"Testing {recognizer.__class__.__name__}")
    
    # Create output directory
    output_dir = Path("output/recognition_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test images directory
    image_dir = Path("data/examples/oneline_images")
    image_paths = list(image_dir.glob("*.jp*g"))
    
    if not image_paths:
        logger.error(f"No test images found in {image_dir}")
        return
    
    # Process each image
    for image_path in image_paths:
        try:
            logger.info(f"\nProcessing {image_path.name}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                continue
            
            # Create debug visualization
            debug_image = image.copy()
            height, width = image.shape[:2]
            
            # Create test bounding box
            margin = int(min(width, height) * 0.1)
            bbox = (margin, margin, width - margin, height - margin)
            
            # Draw bounding box
            cv2.rectangle(debug_image, 
                        (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                        (0, 255, 0), 2)
            
            # Get vehicle attributes
            attributes = recognizer.recognize(image, bbox)
            
            # Draw color sample and info
            color_sample_height = 60
            color_sample = np.zeros((color_sample_height, width, 3), dtype=np.uint8)
            
            # Set sample color
            if attributes.color != 'unknown':
                color_map = {
                    'red': (0, 0, 255),
                    'blue': (255, 0, 0),
                    'green': (0, 255, 0),
                    'white': (255, 255, 255),
                    'black': (0, 0, 0),
                    'silver': (192, 192, 192),
                    'gray': (128, 128, 128),
                    'yellow': (0, 255, 255),
                }
                color_sample[:] = color_map.get(attributes.color.lower(), (128, 128, 128))
            
            # Add text overlay
            text_color = (0, 0, 0)
            if attributes.color.lower() in ['black', 'blue']:
                text_color = (255, 255, 255)
                
            info_text = f"Color: {attributes.color} (conf: {attributes.confidence_scores['color']:.2f})"
            cv2.putText(color_sample, info_text,
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Combine images
            combined_image = np.vstack([debug_image, color_sample])
            
            # Save debug visualization
            debug_path = output_dir / f"debug_{image_path.stem}.jpg"
            cv2.imwrite(str(debug_path), combined_image)
            
            # Log results
            logger.info("Recognition Results:")
            logger.info(f"Color: {attributes.color} ({attributes.confidence_scores['color']:.2f})")
            logger.info(f"Make: {attributes.make} ({attributes.confidence_scores['make']:.2f})")
            logger.info(f"Model: {attributes.model} ({attributes.confidence_scores['model']:.2f})")
            logger.info(f"Type: {attributes.type} ({attributes.confidence_scores['type']:.2f})")
            if attributes.year:
                logger.info(f"Year: {attributes.year}")
            
            logger.info(f"Saved debug image to {debug_path}")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    test_vehicle_recognition()