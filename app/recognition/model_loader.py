# app/recognition/model_loader.py

import torch
import torchvision.models as models
import gdown
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

PRETRAINED_URLS = {
    'car_classifier': 'https://drive.google.com/uc?id=YOUR_GDRIVE_ID',  # Replace with actual URL
    'color_classifier': 'https://drive.google.com/uc?id=YOUR_GDRIVE_ID',  # Replace with actual URL
    'type_classifier': 'https://drive.google.com/uc?id=YOUR_GDRIVE_ID'   # Replace with actual URL
}

def download_weights(model_dir: Path):
    """Download pre-trained weights if they don't exist"""
    os.makedirs(model_dir, exist_ok=True)
    
    for model_name, url in PRETRAINED_URLS.items():
        weight_path = model_dir / f"{model_name}.pth"
        if not weight_path.exists():
            try:
                logger.info(f"Downloading {model_name} weights...")
                gdown.download(url, str(weight_path), quiet=False)
                logger.info(f"Successfully downloaded {model_name} weights")
            except Exception as e:
                logger.error(f"Error downloading {model_name} weights: {str(e)}")

def load_car_classifier(model_dir: Path, device: torch.device) -> torch.nn.Module:
    """Load car make/model classifier"""
    model = models.resnet50(pretrained=True)
    num_classes = 196  # Stanford Cars Dataset classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    weight_path = model_dir / "car_classifier.pth"
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        logger.info("Loaded car classifier weights")
    else:
        logger.warning("No car classifier weights found, using ImageNet weights")
    
    return model.to(device).eval()

def load_color_classifier(model_dir: Path, device: torch.device, num_colors: int) -> torch.nn.Module:
    """Load color classifier"""
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_colors)
    
    weight_path = model_dir / "color_classifier.pth"
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        logger.info("Loaded color classifier weights")
    else:
        logger.warning("No color classifier weights found, using base architecture")
    
    return model.to(device).eval()

def load_type_classifier(model_dir: Path, device: torch.device, num_types: int) -> torch.nn.Module:
    """Load vehicle type classifier"""
    model = models.resnet34(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_types)
    
    weight_path = model_dir / "type_classifier.pth"
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        logger.info("Loaded type classifier weights")
    else:
        logger.warning("No type classifier weights found, using base architecture")
    
    return model.to(device).eval()