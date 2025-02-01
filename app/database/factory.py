# app/database/factory.py

import logging
from typing import Dict, Optional
from flask import current_app
from .timeseries_db import TimeSeriesDB
from .postgres_db import PostgresDB
from .base import DatabaseInterface

logger = logging.getLogger(__name__)

class DatabaseFactory:
    """Factory for creating and managing database connections"""
    
    _instances: Dict[str, DatabaseInterface] = {}
    
    @classmethod
    def get_database(cls, db_type: str) -> Optional[DatabaseInterface]:
        """Get database instance by type"""
        try:
            if db_type not in cls._instances:
                logger.info(f"Creating new database instance for {db_type}")
                
                if db_type == 'timeseries':
                    db = TimeSeriesDB(
                        url=current_app.config['INFLUXDB_URL'],
                        token=current_app.config['INFLUXDB_TOKEN'],
                        org=current_app.config['INFLUXDB_ORG'],
                        bucket=current_app.config['INFLUXDB_BUCKET']
                    )
                elif db_type == 'postgres':
                    db = PostgresDB(
                        dbname=current_app.config['POSTGRES_DB'],
                        user=current_app.config['POSTGRES_USER'],
                        password=current_app.config['POSTGRES_PASSWORD'],
                        host=current_app.config['POSTGRES_HOST'],
                        port=current_app.config['POSTGRES_PORT']
                    )
                else:
                    raise ValueError(f"Unknown database type: {db_type}")
                
                # Test connection
                db.connect()
                cls._instances[db_type] = db
                logger.info(f"Successfully initialized {db_type} database")
                
            return cls._instances[db_type]
            
        except Exception as e:
            logger.error(f"Error getting database {db_type}: {str(e)}")
            return None

    @classmethod
    def get_all_databases(cls) -> Dict[str, DatabaseInterface]:
        """Initialize and get all configured databases"""
        try:
            # Initialize both databases
            timeseries_db = cls.get_database('timeseries')
            postgres_db = cls.get_database('postgres')
            
            if not timeseries_db or not postgres_db:
                raise RuntimeError("Failed to initialize all required databases")
            
            return {
                'timeseries': timeseries_db,
                'postgres': postgres_db
            }
            
        except Exception as e:
            logger.error(f"Error initializing databases: {str(e)}")
            return {}

    @classmethod
    def close_all(cls) -> None:
        """Close all database connections"""
        for db_type, db in cls._instances.items():
            try:
                db.disconnect()
                logger.info(f"Closed {db_type} database connection")
            except Exception as e:
                logger.error(f"Error closing {db_type} database: {str(e)}")
        cls._instances.clear()