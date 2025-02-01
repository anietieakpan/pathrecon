# config/__init__.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class"""
    
    @staticmethod
    def build_database_url(user, password, host, port, db):
        """Build PostgreSQL database URL"""
        return f'postgresql://{user}:{password}@{host}:{port}/{db}'
    
    @staticmethod
    def require_env(key: str) -> str:
        """Get a required environment variable or raise an error"""
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Missing required environment variable: {key}")
        return value
    
    # Flask Configuration
    SECRET_KEY = require_env('SECRET_KEY')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH'))
    
    # PostgreSQL Configuration
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'license_plate_user')
    POSTGRES_PASSWORD = require_env('POSTGRES_PASSWORD')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'license_plate_analysis')
    
    # SQLAlchemy Configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI') or build_database_url(
        POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS', 'False').lower() == 'true'
    SQLALCHEMY_ECHO = os.getenv('SQLALCHEMY_ECHO', 'False').lower() == 'true'
    SQLALCHEMY_POOL_SIZE = int(os.getenv('SQLALCHEMY_POOL_SIZE', 5))
    SQLALCHEMY_MAX_OVERFLOW = int(os.getenv('SQLALCHEMY_MAX_OVERFLOW', 10))
    SQLALCHEMY_POOL_TIMEOUT = int(os.getenv('SQLALCHEMY_POOL_TIMEOUT', 30))
    
    # InfluxDB Configuration
    INFLUXDB_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    INFLUXDB_TOKEN = require_env('INFLUXDB_TOKEN')
    INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'ketu-ai')
    INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', 'license_plate_detections')
    
    # Detection Configuration
    FRAME_SKIP = int(os.getenv('FRAME_SKIP', 2))
    RESIZE_WIDTH = int(os.getenv('RESIZE_WIDTH', 640))
    RESIZE_HEIGHT = int(os.getenv('RESIZE_HEIGHT', 480))
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
    MAX_DETECTIONS_PER_FRAME = int(os.getenv('MAX_DETECTIONS_PER_FRAME', 5))
    PROCESS_EVERY_N_SECONDS = float(os.getenv('PROCESS_EVERY_N_SECONDS', 1))