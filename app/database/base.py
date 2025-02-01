# app/database/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseInterface(ABC):
    """Abstract base class for database interfaces"""
    
    def __init__(self):
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
        
    @abstractmethod
    def connect(self) -> None:
        """Establish database connection"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection"""
        pass
    
    # def ensure_connection(self) -> None:
    def ensure_connection(self):
        """Ensure database is connected"""
        if not self.is_connected:
            self.connect()
    
    def insert_detection(self, detection_data: Dict[str, Any]) -> None:
        """Insert detection data into database"""
        # self.ensure_connection()
        # self._insert_detection_impl(detection_data)
        pass
        

    # def insert_detection(self, detection_data: Dict[str, Any]) -> None:
    #     """Insert detection data into database"""
    #     self.ensure_connection()
    #     self._insert_detection_impl(detection_data)
    
    
    @abstractmethod
    def _insert_detection_impl(self, detection_data: Dict[str, Any]) -> None:
        """Implementation of detection insertion"""
        pass
    
   