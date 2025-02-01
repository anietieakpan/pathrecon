# tests/test_database.py

import pytest
from datetime import datetime, timedelta
from app.database.factory import DatabaseFactory

def test_timeseries_db_insert_and_retrieve():
    db = DatabaseFactory.get_database('timeseries')
    db.connect()

    # Insert a test detection
    test_detection = {
        'text': 'TEST123',
        'confidence': 0.95,
        'timestamp': datetime.now()
    }
    db.insert_detection(test_detection)

    # Retrieve recent detections
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5)
    detections = db.get_detections(start_time, end_time)

    # Verify that our test detection is in the retrieved data
    assert any(d['text'] == 'TEST123' for d in detections)

    db.disconnect()