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
        if len(objects) == 0:
            class_counts['No Object Detected'] = 1
        else:
            for class_label, _ in objects:
                class_counts[class_label] = class_counts.get(class_label, 0) + 1
        
        return JsonResponse({
            'success': True,
            'detections': detections,
            'total_count': len(detections) if len(detections) > 0 else 0,
            'class_counts': class_counts,
            'result_image': result_base64
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def api_detect_frame(request):
    """API endpoint for webcam frame detection"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        # Get frame data from request body
        data = json.loads(request.body)
        frame_data = data.get('frame')
        
        if not frame_data:
            return JsonResponse({'error': 'No frame provided'}, status=400)
        
        # Decode base64 frame
        if 'base64,' in frame_data:
            frame_data = frame_data.split('base64,')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({'error': 'Invalid frame'}, status=400)
        
        # Run detection
        detections, objects = detector.detect_image(frame)
        
        # Draw detections on frame
        result_frame = detector.draw_detections(frame, detections)
        result_base64 = detector.image_to_base64(result_frame)
        
        # Count by class
        class_counts = {}
        if len(objects) == 0:
            class_counts['No Object Detected'] = 1
        else:
            for class_label, _ in objects:
                class_counts[class_label] = class_counts.get(class_label, 0) + 1
        
        return JsonResponse({
            'success': True,
            'detections': detections,
            'total_count': len(detections) if len(detections) > 0 else 0,
            'class_counts': class_counts,
            'result_frame': result_base64
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
