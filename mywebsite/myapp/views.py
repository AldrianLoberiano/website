from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import json
import base64
from .detection.detector import FruitDetector

# Initialize detector
detector = FruitDetector()

def home(request):
    """Home page view"""
    return render(request, 'index.html')

def detect(request):
    """Fruit detection page view"""
    return render(request, 'detect.html')

def video_detect(request):
    """Video detection page view"""
    return render(request, 'video_detect.html')

def history(request):
    """Detection history page view"""
    return render(request, 'history.html')

@csrf_exempt
def api_detect_image(request):
    """API endpoint for image detection"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        # Get image from request
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            image_bytes = image_file.read()
        elif 'image_data' in request.POST:
            # Base64 encoded image
            image_data = request.POST['image_data']
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JsonResponse({'error': 'Invalid image'}, status=400)
        
        # Run detection
        detections, objects = detector.detect_image(image)
        
        # Draw detections on image
        result_image = detector.draw_detections(image, detections)
        result_base64 = detector.image_to_base64(result_image)
        
        # Count by class
        class_counts = {}
        for class_label, _ in objects:
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
        
        return JsonResponse({
            'success': True,
            'detections': detections,
            'total_count': len(detections),
            'class_counts': class_counts,
            'result_image': result_base64
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
