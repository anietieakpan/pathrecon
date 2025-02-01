# from app import create_app

# app = create_app()

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


# run.py

import os
import logging
from app import create_app
from config import Config


# Add to run.py at the start

# import logging

# Configure logging
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/detection.log'),
        logging.StreamHandler()
    ]
)





# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__)

def init_directories():
    """Initialize required directories"""
    directories = [
        'uploads',
        'data/vehicle_images',
        'app/static/css',
        'app/static/js',
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Initialized directory: {directory}")

def main():
    """Main application entry point"""
    try:
        # Initialize required directories
        init_directories()
        
        # Create Flask app
        app = create_app(Config)
        
        # Set debug and testing configs
        debug = os.environ.get('FLASK_DEBUG', '0') == '1'
        testing = os.environ.get('FLASK_TESTING', '0') == '1'
        
        # Additional startup checks
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            logger.error(f"Upload folder not found: {app.config['UPLOAD_FOLDER']}")
            raise RuntimeError("Upload folder not found")
            
        if not hasattr(app, 'databases'):
            logger.warning("No databases initialized on app")
        
        # Run the app
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting application on port {port}")
        app.run(host='0.0.0.0', port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()