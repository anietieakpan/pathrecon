# app/detection/routes.py

from flask import render_template, Response, jsonify, request, current_app
import cv2
import numpy as np
import traceback
import os
from datetime import datetime, timedelta
import pytz
import logging
from werkzeug.utils import secure_filename
from app.detection import bp
from app.detection.detector import LicensePlateDetector
from app.database.factory import DatabaseFactory

logger = logging.getLogger(__name__)

def get_detector():
    """Get or create detector instance"""
    if 'detector' not in current_app.extensions:
        detector = LicensePlateDetector(DatabaseFactory)
        detector.initialize_databases()
        current_app.extensions['detector'] = detector
    return current_app.extensions['detector']

def get_db():
    """Get database instance with error handling"""
    try:
        db = DatabaseFactory.get_database('postgres')
        if db is None:
            raise RuntimeError("Failed to get database instance")
        return db
    except Exception as e:
        logger.error(f"Error getting database: {str(e)}")
        raise

@bp.errorhandler(Exception)
def handle_error(error):
    """Global error handler for the blueprint"""
    logger.error(f"Error in detection routes: {str(error)}", exc_info=True)
    return jsonify({
        'status': 'error',
        'message': str(error)
    }), 500

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(current_app._get_current_object()), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/camera_feed')
def camera_feed():
    detector = get_detector()
    def generate():
        while detector.is_processing:
            frame = detector.get_camera_frame()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @bp.route('/api/vehicle/makes')
# def get_vehicle_makes():
#     """Get hierarchical list of makes with their models"""
#     try:
#         db = get_db()
#         makes = db.get_vehicle_makes_and_models()
        
#         return jsonify({
#             'status': 'success',
#             'makes': makes
#         })
#     except Exception as e:
#         logger.error(f"Error getting vehicle makes: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 500

@bp.route('/api/vehicle/colors')
def get_vehicle_colors():
    """Get list of detected vehicle colors"""
    try:
        db = get_db()
        colors = db.get_vehicle_colors()
        
        return jsonify({
            'status': 'success',
            'colors': colors
        })
    except Exception as e:
        logger.error(f"Error getting vehicle colors: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route('/api/vehicle/search')
def search_vehicles():
    """Search vehicles based on criteria"""
    try:
        # Get search parameters
        make = request.args.get('make')
        model = request.args.get('model')
        color = request.args.get('color')
        type_ = request.args.get('type')
        days = request.args.get('days', default=30, type=int)
        
        # Calculate time range
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=days)
        
        # Build query conditions
        conditions = []
        params = {'start_time': start_time, 'end_time': end_time}
        
        if make:
            conditions.append("vehicle_make = :make")
            params['make'] = make
        if model:
            conditions.append("vehicle_model = :model")
            params['model'] = model
        if color:
            conditions.append("vehicle_color = :color")
            params['color'] = color
        if type_:
            conditions.append("vehicle_type = :type")
            params['type'] = type_
            
        db = get_db()
        vehicles = db.search_vehicles(conditions, params)
        
        return jsonify({
            'status': 'success',
            'vehicles': vehicles
        })
    except Exception as e:
        logger.error(f"Error searching vehicles: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': 'Video uploaded successfully', 'filepath': filepath})

@bp.route('/start_video', methods=['POST'])
def start_video():
    try:
        video_path = request.json.get('videoPath')
        if not video_path:
            return jsonify({'error': 'No video path provided'}), 400
        
        detector = get_detector()
        detector.start_video_capture(video_path)
        return jsonify({'success': 'Video started'})
    except Exception as e:
        return jsonify({'error': f'Error starting video: {str(e)}'}), 500

@bp.route('/stop_video')
def stop_video():
    detector = get_detector()
    detector.stop_video_capture()
    return jsonify({'success': 'Video stopped'})

@bp.route('/start_camera')
def start_camera():
    try:
        detector = get_detector()
        detector.start_camera_capture()
        return jsonify({'success': 'Camera started'})
    except Exception as e:
        return jsonify({'error': f'Error starting camera: {str(e)}'}), 500

@bp.route('/stop_camera')
def stop_camera():
    detector = get_detector()
    detector.stop_camera_capture()
    return jsonify({'success': 'Camera stopped'})

@bp.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    
    if file:
        try:
            # Read and process image
            image_stream = file.read()
            image_array = np.frombuffer(image_stream, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Failed to decode image'}), 400
            
            detector = get_detector()
            encoded_image, detections = detector.process_image(image)
            
            if encoded_image is None:
                return jsonify({'error': 'Error processing image'}), 500
            
            return jsonify({
                'image': encoded_image,
                'detections': detections
            })
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@bp.route('/update_config', methods=['POST'])
def update_config():
    try:
        data = request.json
        detector = get_detector()
        detector.update_config(data)
        
        # Update app config
        valid_keys = [
            'FRAME_SKIP', 'RESIZE_WIDTH', 'RESIZE_HEIGHT',
            'CONFIDENCE_THRESHOLD', 'MAX_DETECTIONS_PER_FRAME',
            'PROCESS_EVERY_N_SECONDS'
        ]
        
        for key, value in data.items():
            if key in valid_keys:
                current_app.config[key] = value
        
        return jsonify({"message": "Configuration updated successfully"})
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/get_config')
def get_config():
    try:
        config = {
            'FRAME_SKIP': current_app.config['FRAME_SKIP'],
            'RESIZE_WIDTH': current_app.config['RESIZE_WIDTH'],
            'RESIZE_HEIGHT': current_app.config['RESIZE_HEIGHT'],
            'CONFIDENCE_THRESHOLD': current_app.config['CONFIDENCE_THRESHOLD'],
            'MAX_DETECTIONS_PER_FRAME': current_app.config['MAX_DETECTIONS_PER_FRAME'],
            'PROCESS_EVERY_N_SECONDS': current_app.config['PROCESS_EVERY_N_SECONDS']
        }
        return jsonify(config)
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        return jsonify({'error': str(e)}), 500

        
@bp.route('/detected_plates')
def detected_plates():
    """Get all detected plates with vehicle details"""
    try:
        detector = get_detector()
        plates = detector.get_detected_plates()
        return jsonify({'plates': plates})
    except Exception as e:
        logger.error(f"Error getting detected plates: {str(e)}")
        return jsonify({'error': str(e)}), 500

        
        

# Add to app/detection/routes.py

@bp.route('/debug_detection')
def debug_detection():
    """Debug endpoint to test vehicle detection"""
    try:
        detector = get_detector()
        
        # Get frame from video or camera
        if detector.cap and detector.is_processing:
            ret, frame = detector.cap.read()
            if not ret:
                return jsonify({'error': 'Could not get frame'})
                
            # Run vehicle detection only
            vehicle_detections = detector.vehicle_detector.detect_vehicles(frame)
            
            # Draw detections
            debug_frame = frame.copy()
            for det in vehicle_detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(debug_frame, 
                            (x1, y1), (x2, y2),
                            (255, 0, 0),
                            2)
            
            # Save debug frame
            cv2.imwrite('output/debug_vehicle_detection.jpg', debug_frame)
            
            return jsonify({
                'message': f'Found {len(vehicle_detections)} vehicles',
                'detections': [
                    {
                        'class': det['class'],
                        'confidence': float(det['confidence']),
                        'bbox': det['bbox']
                    }
                    for det in vehicle_detections
                ]
            })
        else:
            return jsonify({'error': 'No active video feed'})
            
    except Exception as e:
        logger.error(f"Debug detection error: {str(e)}")
        return jsonify({'error': str(e)})


# @bp.route('/db_info', methods=['GET'])
# def get_db_info():
#     """Get database statistics and information"""
#     try:
#         db = DatabaseFactory.get_database('postgres')
        
#         # Get statistics from database
#         vehicle_stats = db.get_vehicle_statistics()
#         makes_models = db.get_vehicle_makes_and_models()
        
#         return jsonify({
#             'statistics': vehicle_stats,
#             'makes_and_models': makes_models
#         })
#     except Exception as e:
#         logger.error(f"Error getting database info: {str(e)}")
#         return jsonify({'error': str(e)}), 500

        
@bp.route('/db_info', methods=['GET'])
def get_db_info():
    """Get database statistics and information"""
    try:
        # First check if databases are initialized
        if not hasattr(current_app, 'databases'):
            logger.warning("No databases initialized on app")
            return jsonify({
                'status': 'warning',
                'message': 'Database connection not initialized',
                'statistics': {},
                'makes_and_models': {}
            })
        
        db = current_app.databases.get('postgres')
        if not db:
            logger.warning("PostgreSQL database not found")
            return jsonify({
                'status': 'warning',
                'message': 'PostgreSQL database not available',
                'statistics': {},
                'makes_and_models': {}
            })
        
        try:
            # Get statistics
            vehicle_stats = db.get_vehicle_statistics()
        except Exception as e:
            logger.error(f"Error getting vehicle statistics: {str(e)}")
            vehicle_stats = {}
            
        try:
            # Get makes and models
            makes_models = db.get_vehicle_makes_and_models()
        except Exception as e:
            logger.error(f"Error getting makes and models: {str(e)}")
            makes_models = {}
        
        return jsonify({
            'status': 'success',
            'statistics': vehicle_stats,
            'makes_and_models': makes_models
        })
        
    except Exception as e:
        logger.error(f"Error in get_db_info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'statistics': {},
            'makes_and_models': {}
        }), 500





# @bp.route('/vehicle/makes', methods=['GET'])
# def get_vehicle_makes():
#     """Get list of vehicle makes and models"""
#     try:
#         db = DatabaseFactory.get_database('postgres')
#         makes_and_models = db.get_vehicle_makes_and_models()
#         return jsonify(makes_and_models)
#     except Exception as e:
#         logger.error(f"Error getting vehicle makes: {str(e)}")
#         return jsonify({'error': str(e)}), 500




@bp.route('/vehicle/makes', methods=['GET'])
def get_vehicle_makes():
    """Get list of vehicle makes and models"""
    try:
        # Check if we have our mapping file
        model_dir = Path('app/models/vehicle')
        mapping_file = model_dir / 'vehicle_mappings.json'
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mappings = json.load(f)
                return jsonify({
                    'status': 'success',
                    'data': mappings
                })
        else:
            # Return default mappings
            default_mappings = {
                'makes': {
                    "0": "BMW",
                    "1": "Mercedes-Benz",
                    "2": "Audi",
                    "3": "Toyota",
                    "4": "Honda"
                },
                'models': {
                    "BMW": {
                        "0": "3 Series",
                        "1": "5 Series",
                        "2": "7 Series"
                    }
                }
            }
            return jsonify({
                'status': 'warning',
                'message': 'Using default mappings',
                'data': default_mappings
            })
            
    except Exception as e:
        logger.error(f"Error getting vehicle makes: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'data': {}
        }), 500





def generate_frames(app):
    """Generate video frames"""
    with app.app_context():
        detector = get_detector()
        try:
            while detector.is_processing:
                frame = detector.get_frame()
                if frame is None:
                    break
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            detector.stop_video_capture()