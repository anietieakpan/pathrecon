# scripts/setup_vehicle_model.py

import os
import requests
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated ModelSetup class
class ModelSetup:
    def __init__(self):
        self.model_dir = Path('app/models/vehicle')
        self.data_dir = Path('app/data/vehicle')
        
        # Model URLs and files
        self.files = {
            'model': {
                'url': 'http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/googlenet_finetune_web_car_iter_10000.caffemodel',
                'output': 'googlenet_cars.caffemodel'
            },
            'prototxt': {
                'content': self._get_prototxt_content(),
                'output': 'deploy.prototxt'
            },
            'classes': {
                'content': self._get_class_mapping(),
                'output': 'car_classes.json'
            }
        }

    def _get_class_mapping(self):
        """Get CompCars class mapping"""
        # This would be the content of the car_classes.json shown above
        return {
            "makes": {
                "0": "Acura",
                "1": "Audi",
                # ... rest of makes
            },
            "models": {
                "0": {
                    "make": "Acura",
                    "model": "ILX",
                    "years": [2013, 2014, 2015],
                    "type": "Sedan"
                },
                # ... rest of models
            }
            # ... rest of mapping
        }

    def _get_prototxt_content(self):
        """Get GoogLeNet prototxt content"""
        # This would be the content of the deploy.prototxt shown above
        return """
        name: "GoogleNet"
        layer {
          name: "data"
          type: "Input"
          top: "data"
          input_param { shape: { dim: 1 dim: 3 dim: 224 dim: 224 } }
        }
        # ... rest of prototxt content
        """

    def setup(self):
        """Run complete setup process"""
        try:
            logger.info("Starting vehicle model setup...")
            
            # Create directories
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Download and create files
            self._download_model()
            self._create_prototxt()
            self._create_class_mapping()
            self._create_model_info()
            
            logger.info("Vehicle model setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            return False

    def _download_model(self):
        """Download caffemodel file"""
        output_path = self.model_dir / self.files['model']['output']
        
        if not output_path.exists():
            logger.info(f"Downloading model file...")
            try:
                response = requests.get(self.files['model']['url'], stream=True)
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0

                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    print(f"Download progress: {percent:.1f}%", end='\r')
                    print()
                    logger.info(f"Successfully downloaded model to {output_path}")
                else:
                    raise Exception(f"Failed to download model: HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading model: {str(e)}")
                raise
        else:
            logger.info(f"Model file already exists at {output_path}")

    def _create_prototxt(self):
        """Create deploy.prototxt file"""
        prototxt_path = self.model_dir / self.files['prototxt']['output']
        try:
            with open(prototxt_path, 'w') as f:
                f.write(self.files['prototxt']['content'])
            logger.info(f"Created prototxt file at {prototxt_path}")
        except Exception as e:
            logger.error(f"Error creating prototxt file: {str(e)}")
            raise

    def _create_class_mapping(self):
        """Create class mapping file"""
        mapping_path = self.data_dir / self.files['classes']['output']
        try:
            with open(mapping_path, 'w') as f:
                json.dump(self.files['classes']['content'], f, indent=2)
            logger.info(f"Created class mapping file at {mapping_path}")
        except Exception as e:
            logger.error(f"Error creating class mapping file: {str(e)}")
            raise

    def _create_model_info(self):
        """Create model info file"""
        info = {
            'name': 'googlenet_cars',
            'version': 'v1.0',
            'creation_date': str(datetime.now()),
            'model_path': str(self.model_dir / self.files['model']['output']),
            'prototxt_path': str(self.model_dir / self.files['prototxt']['output']),
            'classes_path': str(self.data_dir / self.files['classes']['output']),
            'input_shape': [1, 3, 224, 224],
            'mean_values': [104, 117, 123],
            'scale_factor': 1.0
        }
        
        info_path = self.model_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Created model info file at {info_path}")

if __name__ == '__main__':
    setup = ModelSetup()
    setup.setup()