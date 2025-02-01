# tests/recognition/test_basic_recognizer.py

import unittest
import cv2
import numpy as np
from pathlib import Path
from app.recognition import RecognitionType, VehicleRecognizerFactory
from app.recognition.basic_recognizer import BasicVehicleRecognizer

class TestBasicVehicleRecognizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test resources"""
        cls.recognizer = BasicVehicleRecognizer()
        cls.test_data_dir = Path("tests/test_data")
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test image if not exists
        cls.test_image_path = cls.test_data_dir / "test_vehicle.jpg"
        if not cls.test_image_path.exists():
            # Create a simple test image (red rectangle)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[25:75, 25:75] = [0, 0, 255]  # Red rectangle in BGR
            cv2.imwrite(str(cls.test_image_path), img)

    def test_recognizer_creation(self):
        """Test recognizer creation through factory"""
        recognizer = VehicleRecognizerFactory.create_recognizer(RecognitionType.BASIC)
        self.assertIsInstance(recognizer, BasicVehicleRecognizer)

    def test_color_detection(self):
        """Test color detection"""
        # Load test image
        image = cv2.imread(str(self.test_image_path))
        self.assertIsNotNone(image, "Failed to load test image")
        
        bbox = (25, 25, 75, 75)  # Region with red rectangle
        
        # Get attributes
        attributes = self.recognizer.recognize(image, bbox)
        
        # Check color detection
        self.assertEqual(attributes.color.lower(), 'red')
        self.assertGreater(attributes.confidence_scores['color'], 0.5)

    def test_fallback_behavior(self):
        """Test fallback when processing fails"""
        # Create invalid image (wrong number of channels)
        invalid_image = np.zeros((10, 10), dtype=np.uint8)  
        bbox = (0, 0, 5, 5)
        
        # Should return fallback values without error
        attributes = self.recognizer.recognize(invalid_image, bbox)
        self.assertEqual(attributes.make, "Unknown")
        self.assertEqual(attributes.model, "Unknown")
        self.assertEqual(attributes.color.lower(), "unknown")

    def test_supported_attributes(self):
        """Test attribute support checking"""
        self.assertTrue(self.recognizer.supports_attribute('color'))
        self.assertFalse(self.recognizer.supports_attribute('make'))
        self.assertFalse(self.recognizer.supports_attribute('model'))

    def test_confidence_scores(self):
        """Test confidence score ranges"""
        image = cv2.imread(str(self.test_image_path))
        self.assertIsNotNone(image, "Failed to load test image")
        
        bbox = (25, 25, 75, 75)
        
        attributes = self.recognizer.recognize(image, bbox)
        
        # Check confidence scores
        self.assertGreaterEqual(attributes.confidence_scores['color'], 0.0)
        self.assertLessEqual(attributes.confidence_scores['color'], 1.0)
        
        # Other scores should be 0 in basic recognizer
        self.assertEqual(attributes.confidence_scores['make'], 0.0)
        self.assertEqual(attributes.confidence_scores['model'], 0.0)

    def test_image_saving(self):
        """Test vehicle image saving"""
        image = cv2.imread(str(self.test_image_path))
        self.assertIsNotNone(image, "Failed to load test image")
        
        bbox = (25, 25, 75, 75)
        
        attributes = self.recognizer.recognize(image, bbox)
        
        # Check if image was saved
        self.assertIsNotNone(attributes.image_path)
        self.assertTrue(Path(attributes.image_path).exists())

if __name__ == '__main__':
    unittest.main()