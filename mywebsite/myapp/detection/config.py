"""
Configuration for fruit detection system
"""

# Model settings
MODEL_PATH = 'models/fruit_detector.pth'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Fruit classes
FRUIT_CLASSES = [
    'apple',
    'banana',
    'orange',
    'grape',
    'strawberry',
    'watermelon',
    'pineapple',
    'mango',
    'kiwi',
    'pear'
]

# Fruit emojis for display
FRUIT_EMOJIS = {
    'apple': '🍎',
    'banana': '🍌',
    'orange': '🍊',
    'grape': '🍇',
    'strawberry': '🍓',
    'watermelon': '🍉',
    'pineapple': '🍍',
    'mango': '🥭',
    'kiwi': '🥝',
    'pear': '🍐'
}

# Image settings
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp']
IMAGE_RESIZE_MAX = (1024, 1024)

# Detection settings
MAX_DETECTIONS = 100
MIN_DETECTION_SIZE = 20  # Minimum bounding box size in pixels

# Colors for bounding boxes (BGR format for OpenCV)
BBOX_COLORS = {
    'apple': (0, 0, 255),      # Red
    'banana': (0, 255, 255),   # Yellow
    'orange': (0, 165, 255),   # Orange
    'grape': (128, 0, 128),    # Purple
    'strawberry': (255, 0, 255),  # Pink
    'watermelon': (0, 255, 0), # Green
    'pineapple': (0, 255, 255), # Yellow
    'mango': (0, 165, 255),    # Orange
    'kiwi': (0, 128, 0),       # Dark Green
    'pear': (0, 255, 0)        # Green
}
