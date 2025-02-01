# app/recognition/brand_detector.py

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)

class CarBrandDetector:
    """Detects car brands using template matching and feature analysis"""
    
    def __init__(self):
        self.template_dir = Path('app/models/vehicle/templates')
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load brand templates
        self.templates = self._load_templates()
        
        # Brand-specific feature parameters
        self.brand_features = {
            'BMW': {
                'grille_aspect_ratio': (0.4, 0.6),  # Kidney grille ratio range
                'headlight_shape': 'angular',
                'logo_color': ([0, 0, 150], [80, 80, 255]),  # BMW blue range
                'distinctive_features': ['kidney_grille', 'hood_lines', 'headlight_design']
            },
            'Mercedes-Benz': {
                'grille_aspect_ratio': (0.8, 1.2),  # More square grille
                'headlight_shape': 'curved',
                'logo_color': ([200, 200, 200], [255, 255, 255]),  # Silver
                'distinctive_features': ['star_logo', 'grille_pattern', 'hood_emblem']
            },
            'Audi': {
                'grille_aspect_ratio': (1.5, 2.0),  # Wide single-frame grille
                'headlight_shape': 'geometric',
                'logo_color': ([200, 200, 200], [255, 255, 255]),  # Silver
                'distinctive_features': ['single_frame_grille', 'led_signature', 'four_rings']
            }
        }

    def _load_templates(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load brand-specific templates"""
        templates = {}
        try:
            # Load for each brand
            for brand in ['BMW', 'Mercedes-Benz', 'Audi']:
                brand_dir = self.template_dir / brand.lower()
                if brand_dir.exists():
                    templates[brand] = {}
                    for template_path in brand_dir.glob('*.png'):
                        feature_name = template_path.stem
                        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                        if template is not None:
                            templates[brand][feature_name] = template
            
            logger.info(f"Loaded templates for brands: {list(templates.keys())}")
            return templates
            
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            return {}

    def detect_brand(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """Detect car brand using multiple methods"""
        try:
            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Initialize results
            brand_scores = {
                'BMW': 0.0,
                'Mercedes-Benz': 0.0,
                'Audi': 0.0
            }
            
            # 1. Template Matching
            template_scores = self._template_matching(gray)
            
            # 2. Feature Analysis
            feature_scores = self._analyze_distinctive_features(image)
            
            # 3. Color Analysis
            color_scores = self._analyze_brand_colors(image)
            
            # 4. Shape Analysis
            shape_scores = self._analyze_brand_shapes(gray)
            
            # Combine scores
            for brand in brand_scores:
                brand_scores[brand] = (
                    template_scores.get(brand, 0.0) * 0.4 +
                    feature_scores.get(brand, 0.0) * 0.3 +
                    color_scores.get(brand, 0.0) * 0.2 +
                    shape_scores.get(brand, 0.0) * 0.1
                )
            
            # Get best match
            best_brand = max(brand_scores.items(), key=lambda x: x[1])
            
            # Return if confidence is high enough
            if best_brand[1] > 0.6:
                return best_brand[0], best_brand[1], {
                    'template_score': template_scores.get(best_brand[0], 0.0),
                    'feature_score': feature_scores.get(best_brand[0], 0.0),
                    'color_score': color_scores.get(best_brand[0], 0.0),
                    'shape_score': shape_scores.get(best_brand[0], 0.0)
                }
            
            return "Unknown", 0.0, {}
            
        except Exception as e:
            logger.error(f"Error in brand detection: {str(e)}")
            return "Unknown", 0.0, {}

    def _template_matching(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Match against brand-specific templates"""
        scores = {}
        
        try:
            for brand, templates in self.templates.items():
                brand_score = 0.0
                matches = 0
                
                for feature_name, template in templates.items():
                    # Multi-scale template matching
                    scales = np.linspace(0.5, 1.5, 5)
                    max_val = 0
                    
                    for scale in scales:
                        scaled_template = cv2.resize(template, None, 
                                                   fx=scale, fy=scale)
                        
                        if scaled_template.shape[0] > gray_image.shape[0] or \
                           scaled_template.shape[1] > gray_image.shape[1]:
                            continue
                        
                        result = cv2.matchTemplate(gray_image, scaled_template, 
                                                 cv2.TM_CCOEFF_NORMED)
                        _, val, _, _ = cv2.minMaxLoc(result)
                        max_val = max(max_val, val)
                    
                    if max_val > 0.7:  # Good match threshold
                        brand_score += max_val
                        matches += 1
                
                if matches > 0:
                    scores[brand] = brand_score / matches
                    
            return scores
            
        except Exception as e:
            logger.error(f"Error in template matching: {str(e)}")
            return {}

    def _analyze_distinctive_features(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze brand-specific distinctive features"""
        scores = {}
        
        try:
            # Convert to grayscale and get edges
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            for brand, features in self.brand_features.items():
                score = 0.0
                
                # Check grille aspect ratio
                grille_score = self._detect_grille(edges, features['grille_aspect_ratio'])
                
                # Check headlight shape
                headlight_score = self._detect_headlights(edges, features['headlight_shape'])
                
                # Combine scores
                score = (grille_score * 0.6 + headlight_score * 0.4)
                if score > 0:
                    scores[brand] = score
                    
            return scores
            
        except Exception as e:
            logger.error(f"Error in feature analysis: {str(e)}")
            return {}
        
    def _detect_grille(self, edges: np.ndarray, 
                      aspect_ratio_range: Tuple[float, float]) -> float:
        """Detect and analyze grille shape"""
        try:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            max_score = 0.0
            for contour in contours:
                if len(contour) > 10:  # Minimum points for a grille
                    x, y, w, h = cv2.boundingRect(contour)
                    ratio = w / h if h != 0 else 0
                    
                    if aspect_ratio_range[0] <= ratio <= aspect_ratio_range[1]:
                        area_ratio = cv2.contourArea(contour) / (w * h)
                        if 0.4 <= area_ratio <= 0.8:  # Typical grille area ratio
                            max_score = max(max_score, area_ratio)
                            
            return max_score
            
        except Exception as e:
            logger.error(f"Error in grille detection: {str(e)}")
            return 0.0

    def _detect_headlights(self, edges: np.ndarray, shape_type: str) -> float:
        """Detect and analyze headlight shapes"""
        try:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            max_score = 0.0
            for contour in contours:
                if len(contour) > 10:
                    x, y, w, h = cv2.boundingRect(contour)
                    ratio = w / h if h != 0 else 0
                    
                    if 1.5 <= ratio <= 4.0:  # Typical headlight ratio range
                        if shape_type == 'angular':
                            # Check for angular features
                            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                            if len(approx) >= 6:
                                max_score = max(max_score, 0.8)
                        elif shape_type == 'curved':
                            # Check for curved features
                            area_ratio = cv2.contourArea(contour) / (w * h)
                            if 0.6 <= area_ratio <= 0.9:
                                max_score = max(max_score, 0.8)
                                
            return max_score
            
        except Exception as e:
            logger.error(f"Error in headlight detection: {str(e)}")
            return 0.0

    def _detect_brand_specific_features(self, image: np.ndarray, brand: str) -> float:
        """Detect brand-specific features"""
        try:
            if brand == 'BMW':
                return self._detect_bmw_features(image)
            elif brand == 'Mercedes-Benz':
                return self._detect_mercedes_features(image)
            elif brand == 'Audi':
                return self._detect_audi_features(image)
            return 0.0
        except Exception as e:
            logger.error(f"Error in brand-specific feature detection: {str(e)}")
            return 0.0

    def _detect_bmw_features(self, image: np.ndarray) -> float:
        """Detect BMW-specific features"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Check for kidney grille
            grille_score = self._detect_kidney_grille_pattern(edges)
            
            # Check for BMW badge
            badge_score = self._detect_circular_badge(gray)
            
            # Check for hood lines
            hood_score = self._detect_hood_lines(edges)
            
            return (grille_score * 0.5 + badge_score * 0.3 + hood_score * 0.2)
        except Exception as e:
            logger.error(f"Error in BMW feature detection: {str(e)}")
            return 0.0

    def _detect_mercedes_features(self, image: np.ndarray) -> float:
        """Detect Mercedes-specific features"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Check for star logo
            star_score = self._detect_star_pattern(edges)
            
            # Check for grille pattern
            grille_score = self._detect_mercedes_grille(edges)
            
            # Check for hood emblem
            emblem_score = self._detect_hood_emblem(gray)
            
            return (star_score * 0.5 + grille_score * 0.3 + emblem_score * 0.2)
        except Exception as e:
            logger.error(f"Error in Mercedes feature detection: {str(e)}")
            return 0.0

    def _detect_audi_features(self, image: np.ndarray) -> float:
        """Detect Audi-specific features"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Check for four rings
            rings_score = self._detect_rings_pattern(edges)
            
            # Check for single-frame grille
            grille_score = self._detect_singleframe_grille(edges)
            
            # Check for LED signature
            led_score = self._detect_led_signature(gray)
            
            return (rings_score * 0.5 + grille_score * 0.3 + led_score * 0.2)
        except Exception as e:
            logger.error(f"Error in Audi feature detection: {str(e)}")
            return 0.0