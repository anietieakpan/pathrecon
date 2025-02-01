from flask import Blueprint

bp = Blueprint('detection', __name__)

from app.detection import routes