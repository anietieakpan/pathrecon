from app.api import bp

@bp.route('/detected_plates')
def detected_plates():
    # Implement this route
    pass

- GET /api/detections (with vehicle details)
- GET /api/vehicles/stats (vehicle make/model statistics)
- GET /api/vehicles/search (search by make, model, color)