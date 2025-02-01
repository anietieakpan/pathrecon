# tests/test_vehicle_classifier.py

import unittest
import cv2
import numpy as np
from pathlib import Path
from app.detection.vehicle_classifier import VehicleClassifier

class TestVehicleClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources"""
        cls.classifier = VehicleClassifier()
        
        # Create test image directory if it doesn't exist
        cls.test_images_dir = Path("tests/test_images")
        cls.test_images_dir.mkdir(parents=True, exist_ok=True)

    def test_model_loading(self):
        """Test if model loads correctly"""
        self.assertIsNotNone(self.classifier.model)
        if self.classifier.labels:
            self.assertIsInstance(self.classifier.labels, dict)

    def test_color_detection(self):
        """Test color detection on sample images"""
        # Create sample colored images
        colors_to_test = {
            'white': np.ones((100, 100, 3), dtype=np.uint8) * 255,
            'black': np.zeros((100, 100, 3), dtype=np.uint8),
            'red': np.zeros((100, 100, 3), dtype=np.uint8)
        }
        colors_to_test['red'][:, :, 2] = 255  # BGR format
        
        for expected_color, image in colors_to_test.items():
            color, confidence = self.classifier._detect_color_enhanced(image)
            self.assertEqual(color.lower(), expected_color.lower())
            self.assertGreater(confidence, 0.5)

    def test_image_preprocessing(self):
        """Test image preprocessing"""
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        processed = self.classifier.preprocess_image(test_image)
        
        # Check output shape and normalization
        self.assertEqual(processed.shape[1:3], self.classifier.input_size)
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1))

    def test_vehicle_processing(self):
        """Test full vehicle processing pipeline"""
        # Create a test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        bbox = (50, 50, 250, 350)
        
        result = self.classifier.process_vehicle_image(test_image, bbox)
        
        # Check result structure
        required_fields = ['make', 'model', 'color', 'type', 'confidence_scores']
        for field in required_fields:
            self.assertIn(field, result)
        
        # Check confidence scores
        self.assertIsInstance(result['confidence_scores'], dict)
        for score in result['confidence_scores'].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        # Test with invalid image
        invalid_image = np.zeros((10, 10), dtype=np.uint8)  # Wrong number of channels
        result = self.classifier.process_vehicle_image(invalid_image, (0, 0, 5, 5))
        self.assertEqual(result['make'], 'Unknown')
        
        # Test with invalid bbox
        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.classifier.process_vehicle_image(valid_image, (-10, -10, 150, 150))
        self.assertIsNotNone(result)  # Should return fallback prediction

    def test_image_saving(self):
        """Test vehicle image saving functionality"""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        image_path = self.classifier._save_vehicle_image(test_image)
        
        self.assertIsNotNone(image_path)
        self.assertTrue(Path(image_path).exists())
        
        # Clean up
        Path(image_path).unlink()

if __name__ == '__main__':
    unittest.main()filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': 'Video uploaded successfully', 'filepath': filepath})
