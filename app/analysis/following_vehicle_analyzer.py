from datetime import datetime, timedelta

class FollowingVehicleAnalyzer:
    def __init__(self, database):
        self.database = database

    def analyze_following_vehicles(self, subject_plate, time_window):
        # This is a simplified example. In a real-world scenario, 
        # you'd need a more complex algorithm to determine following vehicles.
        subject_detections = self.database.get_detections(
            start_time=datetime.now() - time_window,
            end_time=datetime.now()
        )
        subject_detections = [d for d in subject_detections if d['text'] == subject_plate]
        subject_detections.sort(key=lambda x: x['timestamp'])

        following_vehicles = []
        for i in range(len(subject_detections) - 1):
            potential_followers = self.database.get_detections(
                start_time=subject_detections[i]['timestamp'],
                end_time=subject_detections[i+1]['timestamp']
            )
            
            for follower in potential_followers:
                if follower['text'] != subject_plate and follower['text'] not in following_vehicles:
                    following_vehicles.append(follower['text'])

        return following_vehicles

    def is_vehicle_following(self, subject_plate, suspect_plate, time_window):
        following_vehicles = self.analyze_following_vehicles(subject_plate, time_window)
        return suspect_plate in following_vehicles