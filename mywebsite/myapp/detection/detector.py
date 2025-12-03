import cv2
import numpy as np
import os
import base64
from pathlib import Path

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Using mock detection.")


class FruitDetector:
    """TensorFlow Lite-based fruit ripeness detector for Django"""
    
    def __init__(self, model_path="eggplant.tflite"):
        self.class_names = ["Unripe", "Semi-Ripe", "Ripe", "Overripe", "Defective"]
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
        self.model_loaded = False
        
        # Try to load the model
        if TF_AVAILABLE and os.path.exists(model_path):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.model_loaded = True
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model_loaded = False
        else:
            print(f"Model not found at {model_path} or TensorFlow not available. Using mock detection.")

    def preprocess_image(self, image, input_size):
        """Preprocess image for model input"""
        image_resized = cv2.resize(image, input_size)
        image_normalized = image_resized / 255.0
        input_data = np.expand_dims(image_normalized, axis=0).astype(np.float32)
        return input_data
    
    def non_max_suppression(self, bboxes, scores, iou_threshold):
        """Apply non-maximum suppression to remove overlapping boxes"""
        indices = cv2.dnn.NMSBoxes(bboxes, scores, self.confidence_threshold, iou_threshold)
        return indices.flatten() if len(indices) > 0 else []





    def postprocess_output(self, detections, image):
        """Process model output and extract bounding boxes"""
        height, width, _ = image.shape
        boxes, scores, labels = [], [], []

        for detection in detections:
            score = detection[4]
            if score >= self.confidence_threshold:
                x_center, y_center, w, h = detection[0:4]
                x1 = int((x_center - w / 2) * width)
                y1 = int((y_center - h / 2) * height)
                x2 = int((x_center + w / 2) * width)
                y2 = int((y_center + h / 2) * height)

                class_probs = detection[5:]
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]

                if class_score >= self.confidence_threshold:
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    scores.append(float(score))
                    labels.append(class_id)

        indices = self.non_max_suppression(boxes, scores, self.iou_threshold)
        bounding_boxes = []
        detected_objects = []

        for idx in indices:
            x1, y1, w, h = boxes[idx]
            x2, y2 = x1 + w, y1 + h
            
            class_id = labels[idx]
            if class_id >= len(self.class_names):
                continue
                
            class_label = self.class_names[class_id]
            confidence = scores[idx]
            
            bounding_boxes.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'class': class_label,
                'confidence': float(confidence)
            })
            detected_objects.append((class_label, confidence))

        return bounding_boxes, detected_objects


    def detect_image(self, image):
        """Run detection on a single image"""
        if not self.model_loaded:
            return self._mock_detection(image)
        
        try:
            input_size = (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1])
            input_data = self.preprocess_image(image, input_size)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            bounding_boxes, detected_objects = self.postprocess_output(output_data, image)
            return bounding_boxes, detected_objects
        except Exception as e:
            print(f"Detection error: {e}")
            return self._mock_detection(image)
    
    def _mock_detection(self, image):
        """Generate mock detection results for testing"""
        height, width = image.shape[:2]
        detections = []
        objects = []
        
        # Create 2-4 random detections
        num_detections = np.random.randint(2, 5)
        for i in range(num_detections):
            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(0, height - 100)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(50, 150)
            
            class_label = np.random.choice(self.class_names)
            confidence = np.random.uniform(0.6, 0.95)
            
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'class': class_label,
                'confidence': float(confidence)
            })
            objects.append((class_label, confidence))
        
        return detections, objects
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        result_image = image.copy()
        
        color_map = {
            'Unripe': (0, 0, 255),      # Red
            'Semi-Ripe': (0, 165, 255),  # Orange
            'Ripe': (0, 255, 0),         # Green
            'Overripe': (0, 255, 255),   # Yellow
            'Defective': (128, 0, 128)   # Purple
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            class_label = det['class']
            confidence = det['confidence']
            
            color = color_map.get(class_label, (0, 255, 0))
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_label}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def image_to_base64(self, image):
        """Convert image to base64 string for web display"""
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"