# app/database/distribution/core.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import queue
import threading
from dataclasses import dataclass

@dataclass
class DetectionEvent:
    """Represents a license plate detection event"""
    plate_text: str
    confidence: float
    timestamp_utc: datetime
    timestamp_local: datetime
    location: Optional[Dict[str, float]] = None  # For future GPS integration
    camera_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DataConsumer(ABC):
    """Abstract base class for database-specific consumers"""
    
    @abstractmethod
    def process_detection(self, detection: DetectionEvent) -> bool:
        """Process a single detection event"""
        pass
    
    @abstractmethod
    def process_batch(self, detections: List[DetectionEvent]) -> bool:
        """Process a batch of detection events"""
        pass
    
    @abstractmethod
    def get_last_processed_time(self) -> datetime:
        """Get the timestamp of the last processed event"""
        pass

class DataDistributor:
    """Manages data distribution to various consumer databases"""
    
    def __init__(self, batch_size: int = 100, processing_interval: float = 1.0):
        self.consumers: Dict[str, DataConsumer] = {}
        self.queues: Dict[str, queue.Queue] = {}
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.is_running = False
        self.threads: Dict[str, threading.Thread] = {}
        self.logger = logging.getLogger(__name__)

    def register_consumer(self, name: str, consumer: DataConsumer):
        """Register a new database consumer"""
        self.consumers[name] = consumer
        self.queues[name] = queue.Queue()
        self.logger.info(f"Registered new consumer: {name}")

    def distribute_event(self, detection: DetectionEvent):
        """Distribute a detection event to all consumers"""
        for name, q in self.queues.items():
            q.put(detection)
            self.logger.debug(f"Queued detection {detection.plate_text} for {name}")

    def start(self):
        """Start processing threads for all consumers"""
        self.is_running = True
        for name, consumer in self.consumers.items():
            thread = threading.Thread(
                target=self._process_queue,
                args=(name, consumer),
                name=f"distributor-{name}"
            )
            thread.daemon = True
            thread.start()
            self.threads[name] = thread
            self.logger.info(f"Started processing thread for {name}")

    def stop(self):
        """Stop all processing threads"""
        self.is_running = False
        for thread in self.threads.values():
            thread.join()
        self.logger.info("Stopped all processing threads")

    def _process_queue(self, name: str, consumer: DataConsumer):
        """Process queue for a specific consumer"""
        queue = self.queues[name]
        batch = []
        
        while self.is_running:
            try:
                # Collect batch of events
                while len(batch) < self.batch_size:
                    try:
                        detection = queue.get_nowait()
                        batch.append(detection)
                    except queue.Empty:
                        break

                # Process batch if we have any events
                if batch:
                    try:
                        success = consumer.process_batch(batch)
                        if not success:
                            self.logger.error(f"Failed to process batch for {name}")
                            # Requeue failed batch
                            for detection in batch:
                                queue.put(detection)
                    except Exception as e:
                        self.logger.error(f"Error processing batch for {name}: {str(e)}")
                        # Requeue failed batch
                        for detection in batch:
                            queue.put(detection)
                    finally:
                        batch = []

                # Sleep if queue is empty
                if queue.empty():
                    threading.Event().wait(self.processing_interval)
                    
            except Exception as e:
                self.logger.error(f"Error in processing thread for {name}: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all consumers"""
        return {
            name: {
                'queue_size': self.queues[name].qsize(),
                'last_processed': consumer.get_last_processed_time(),
                'is_running': name in self.threads and self.threads[name].is_alive()
            }
            for name, consumer in self.consumers.items()
        }