import numpy as np
import cv2
from ultralytics import YOLO
import torch
import json

class MechanicalKeyboardDetector:
    def __init__(self, model_path='best.pt'):
        """
        Initialize detector with custom trained model for mechanical keyboards
        Args:
            model_path: Path to trained YOLOv8 model weights
        """
        self.TOTAL_LENGTH_MM = 354.076
        self.TOTAL_WIDTH_MM = 123.444
        
        # Load custom trained YOLO model
        try:
            self.model = YOLO(model_path)
        except:
            print("Custom model not found, using base YOLOv8...")
            self.model = YOLO('yolov8n.pt')
            
        # Load key positions
        self.key_positions = {}
        with open('final_points.json', 'r') as f:
                self.key_positions = {
                key: np.array(pos, dtype=np.float32)
                for key, pos in json.load(f).items()
                }
        
    def load_key_positions(self):
        """Load mechanical keyboard key positions"""
        key_data = {
            "ESC": [9.64, 8.4225],
            "F1": [49.21, 8.4225],
            # ... rest of your JSON data
        }
        return {k: np.array(v, dtype=np.float32) for k, v in key_data.items()}
    
    def detect_keyboard(self, image, conf_threshold=0.3):
        """
        Detect mechanical keyboard in image
        Args:
            image: Input BGR image
            conf_threshold: Confidence threshold for detection
        Returns:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            confidence: Detection confidence
        """
        # Run inference
        results = self.model(image)
        
        # Filter for keyboard detections
        keyboard_detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # For custom model, class 0 would be mechanical keyboard
                # For base model, class 63 is keyboard
                if (box.cls == 0 or box.cls == 63) and box.conf > conf_threshold:
                    keyboard_detections.append((box.xyxy[0].cpu().numpy(), box.conf.cpu().numpy()))
        
        if not keyboard_detections:
            return None, None
            
        # Sort by confidence and get best detection
        keyboard_detections.sort(key=lambda x: x[1], reverse=True)
        best_bbox, best_conf = keyboard_detections[0]
        
        return best_bbox, best_conf
    
    def extract_keyboard_features(self, image, bbox):
        """
        Extract features specific to mechanical keyboards
        Args:
            image: Input image
            bbox: Bounding box coordinates
        Returns:
            is_mechanical: Boolean indicating if keyboard appears mechanical
        """
        x1, y1, x2, y2 = map(int, bbox)
        keyboard_roi = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(keyboard_roi, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find key boundaries
        edges = cv2.Canny(gray, 100, 200)
        
        # Count edges - mechanical keyboards typically have more defined edges
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Mechanical keyboards typically have higher edge density
        return edge_density > 0.1
    
    def process_image(self, image):
        """
        Process image to detect mechanical keyboard and validate
        Args:
            image: Input BGR image
        Returns:
            result_dict: Dictionary containing detection results
        """
        # Detect keyboard
        bbox, confidence = self.detect_keyboard(image)
        
        if bbox is None:
            return {
                'detected': False,
                'message': 'No keyboard detected'
            }
            
        # Validate if mechanical
        is_mechanical = self.extract_keyboard_features(image, bbox)
        
        return {
            'detected': True,
            'bbox': bbox.tolist(),
            'confidence': float(confidence),
            'is_mechanical': is_mechanical
        }
    
    def draw_results(self, image, results):
        """Draw detection results on image"""
        if not results['detected']:
            return image
            
        result = image.copy()
        bbox = results['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        color = (0, 255, 0) if results['is_mechanical'] else (0, 0, 255)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Add labels
        label = f"Mechanical Keyboard: {results['confidence']:.2f}"
        cv2.putText(result, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result

def main():
    # Initialize detector
    detector = MechanicalKeyboardDetector()
    
    # Initialize image capture by reading image '266.jpg'
    image = cv2.imread('26.jpg')
    # Process image
    results = detector.process_image(image)
    
    # Draw results
    output = detector.draw_results(image, results)
    
    # Display
    cv2.imshow('Mechanical Keyboard Detection', output)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()