"""
Utility functions for fruit detection
"""
import os
from pathlib import Path
from PIL import Image
import time


def validate_image(image_path):
    """
    Validate if the file is a valid image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False


def get_image_info(image_path):
    """
    Get basic information about an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Image information (size, format, mode)
    """
    img = Image.open(image_path)
    return {
        'size': img.size,
        'format': img.format,
        'mode': img.mode,
        'width': img.width,
        'height': img.height
    }


def resize_image(image_path, max_size=(800, 800), output_path=None):
    """
    Resize image maintaining aspect ratio
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimensions (width, height)
        output_path: Where to save resized image (optional)
        
    Returns:
        str: Path to the resized image
    """
    img = Image.open(image_path)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    if output_path is None:
        output_path = image_path
    
    img.save(output_path)
    return output_path


def create_upload_directory(base_dir='media/uploads'):
    """
    Create directory for uploaded images
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Path: Created directory path
    """
    upload_dir = Path(base_dir) / time.strftime('%Y%m%d')
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def save_uploaded_file(uploaded_file, directory):
    """
    Save an uploaded file to the specified directory
    
    Args:
        uploaded_file: Django UploadedFile object
        directory: Directory to save the file
        
    Returns:
        Path: Path to the saved file
    """
    filename = uploaded_file.name
    filepath = Path(directory) / filename
    
    with open(filepath, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    return filepath
