from django.shortcuts import render

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
