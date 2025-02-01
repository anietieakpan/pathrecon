# from app import db
# from datetime import datetime

# class Detection(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     license_plate = db.Column(db.String(20), index=True)
#     confidence = db.Column(db.Float)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     image_path = db.Column(db.String(255))

#     def __repr__(self):
#         return f'<Detection {self.license_plate}>'