# app/detection/model_validator.py

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class ModelValidator:
    """Validates vehicle detection model setup and files"""
    
    def __init__(self, model_dir: str = 'app/models/vehicle', data_dir: str = 'app/data/vehicle'):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.required_files = [
            'googlenet_cars.caffemodel',
            'deploy.prototxt',
            'model_info.json'
        ]
        self.required_data_files = [
            'car_classes.json'
        ]

    def validate_setup(self) -> Tuple[bool, List[str]]:
        """Validate complete model setup"""
        errors = []
        
        try:
            # Check directories
            if not self._check_directories():
                errors.append("Required directories are missing")
                return False, errors
            
            # Check files
            missing_files = self._check_files()
            if missing_files:
                errors.extend(f"Missing file: {f}" for f in missing_files)
            
            # Validate model info
            info_valid, info_errors = self._validate_model_info()
            if not info_valid:
                errors.extend(info_errors)
            
            # Validate class mapping
            mapping_valid, mapping_errors = self._validate_class_mapping()
            if not mapping_valid:
                errors.extend(mapping_errors)
            
            # Test model loading
            load_valid, load_errors = self._test_model_loading()
            if not load_valid:
                errors.extend(load_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors

    def _check_directories(self) -> bool:
        """Check if required directories exist"""
        return self.model_dir.exists() and self.data_dir.exists()

    def _check_files(self) -> List[str]:
        """Check if required files exist"""
        missing_files = []
        
        for file in self.required_files:
            if not (self.model_dir / file).exists():
                missing_files.append(file)
                
        for file in self.required_data_files:
            if not (self.data_dir / file).exists():
                missing_files.append(file)
                
        return missing_files

    def _validate_model_info(self) -> Tuple[bool, List[str]]:
        """Validate model info file"""
        errors = []
        try:
            info_path = self.model_dir / 'model_info.json'
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            required_fields = [
                'name', 'version', 'model_path', 'prototxt_path',
                'classes_path', 'input_shape', 'mean_values'
            ]
            
            for field in required_fields:
                if field not in info:
                    errors.append(f"Missing field in model_info.json: {field}")
            
            # Validate paths in info
            model_path = Path(info.get('model_path', ''))
            prototxt_path = Path(info.get('prototxt_path', ''))
            classes_path = Path(info.get('classes_path', ''))
            
            if not model_path.exists():
                errors.append(f"Model file not found: {model_path}")
            if not prototxt_path.exists():
                errors.append(f"Prototxt file not found: {prototxt_path}")
            if not classes_path.exists():
                errors.append(f"Classes file not found: {classes_path}")
                
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Error validating model info: {str(e)}"]

    def _validate_class_mapping(self) -> Tuple[bool, List[str]]:
        """Validate class mapping file"""
        errors = []
        try:
            mapping_path = self.data_dir / 'car_classes.json'
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            
            required_sections = ['makes', 'models', 'types', 'years']
            for section in required_sections:
                if section not in mapping:
                    errors.append(f"Missing section in class mapping: {section}")
            
            if 'models' in mapping:
                # Validate model entries
                for idx, model in mapping['models'].items():
                    required_fields = ['make', 'model', 'years', 'type']
                    for field in required_fields:
                        if field not in model:
                            errors.append(f"Missing field '{field}' in model {idx}")
                    
                    # Check if make exists in makes section
                    if 'makes' in mapping and model['make'] not in mapping['makes'].values():
                        errors.append(f"Invalid make '{model['make']}' in model {idx}")
                    
                    # Check if type exists in types section
                    if 'types' in mapping and model['type'] not in mapping['types'].values():
                        errors.append(f"Invalid type '{model['type']}' in model {idx}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Error validating class mapping: {str(e)}"]

    def _test_model_loading(self) -> Tuple[bool, List[str]]:
        """Test if model can be loaded"""
        errors = []
        try:
            # Get model paths
            with open(self.model_dir / 'model_info.json', 'r') as f:
                info = json.load(f)
            
            model_path = info['model_path']
            prototxt_path = info['prototxt_path']
            
            # Try loading model
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            # Test with dummy input
            dummy_input = np.random.random((1, 3, 224, 224)).astype(np.float32)
            net.setInput(dummy_input)
            
            # Try getting output
            output = net.forward()
            
            if output.shape[1] != 431:  # Number of classes in CompCars
                errors.append(f"Unexpected model output shape: {output.shape}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Error testing model: {str(e)}"]

    def validate_inference(self, test_image_path: str) -> Tuple[bool, List[str], Dict]:
        """Validate model inference with a test image"""
        errors = []
        results = {}
        
        try:
            # Load test image
            image = cv2.imread(test_image_path)
            if image is None:
                return False, ["Failed to load test image"], {}
            
            # Load model info
            with open(self.model_dir / 'model_info.json', 'r') as f:
                info = json.load(f)
            
            # Load class mapping
            with open(self.data_dir / 'car_classes.json', 'r') as f:
                class_mapping = json.load(f)
            
            # Prepare image
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (224, 224),
                mean=tuple(info['mean_values']),
                swapRB=True
            )
            
            # Load and run model
            net = cv2.dnn.readNetFromCaffe(info['prototxt_path'], info['model_path'])
            net.setInput(blob)
            output = net.forward()
            
            # Get top predictions
            top_indices = output[0].argsort()[-5:][::-1]
            predictions = []
            
            for idx in top_indices:
                confidence = float(output[0][idx])
                if idx < len(class_mapping['models']):
                    model_info = class_mapping['models'][str(idx)]
                    predictions.append({
                        'make': model_info['make'],
                        'model': model_info['model'],
                        'type': model_info['type'],
                        'confidence': confidence
                    })
            
            if not predictions:
                errors.append("No valid predictions generated")
            
            results = {
                'predictions': predictions,
                'inference_time': None,  # Could add timing information
                'image_size': image.shape
            }
            
            return len(errors) == 0, errors, results
            
        except Exception as e:
            return False, [f"Error during inference validation: {str(e)}"], {}

    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'opencv_version': cv2.__version__,
                'numpy_version': np.__version__,
                'cuda_available': cv2.cuda.getCudaEnabledDeviceCount() > 0,
                'opencl_available': cv2.ocl.haveOpenCL()
            },
            'model_info': {},
            'validation_results': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Run all validations
        setup_valid, setup_errors = self.validate_setup()
        report['validation_results']['setup'] = {
            'status': 'pass' if setup_valid else 'fail',
            'errors': setup_errors
        }
        
        # Add model information
        try:
            with open(self.model_dir / 'model_info.json', 'r') as f:
                report['model_info'] = json.load(f)
        except Exception as e:
            report['warnings'].append(f"Could not load model info: {str(e)}")
        
        # Add recommendations based on validation results
        if not setup_valid:
            report['recommendations'].append("Fix model setup issues before using the model")
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            report['recommendations'].append("CUDA is available - model will use GPU acceleration")
        elif cv2.ocl.haveOpenCL():
            report['recommendations'].append("OpenCL is available - model will use limited GPU acceleration")
        else:
            report['recommendations'].append("No GPU acceleration available - consider enabling CUDA or OpenCL")
        
        return report

def run_validation(test_image: str = None):
    """Run validation from command line"""
    validator = ModelValidator()
    success, errors = validator.validate_setup()
    
    print("Model Validation Results:")
    print("-" * 50)
    
    if success:
        print("✓ Basic setup validation passed")
    else:
        print("✗ Setup validation failed:")
        for error in errors:
            print(f"  - {error}")
    
    if test_image:
        print("\nRunning inference test:")
        success, errors, results = validator.validate_inference(test_image)
        if success:
            print("✓ Inference test passed")
            print("\nTop predictions:")
            for pred in results['predictions'][:3]:
                print(f"  {pred['make']} {pred['model']}: {pred['confidence']:.2%}")
        else:
            print("✗ Inference test failed:")
            for error in errors:
                print(f"  - {error}")
    
    report = validator.generate_validation_report()
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

if __name__ == '__main__':
    import sys
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    run_validation(test_image)