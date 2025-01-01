import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from torchvision.ops import nms

class AdvancedKeyboardDetector:
    def __init__(self, model_path='yolov8x-seg.pt'):
        """
        Initialize with YOLOv8 segmentation model for more precise initial detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize background subtractor for motion analysis
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
            
    def enhance_image(self, image):
        """
        Advanced image enhancement pipeline
        """
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Apply bilateral filter for edge-preserving smoothing
        l_smooth = cv2.bilateralFilter(l_enhanced, 9, 75, 75)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_smooth, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # # Apply guided filter for detail preservation
        # enhanced = cv2.ximgproc.guidedFilter(
        #     guide=enhanced, src=enhanced, radius=3, eps=1e-4)
            
        return enhanced
        
    def extract_features(self, image, mask):
        """
        Extract advanced features from the keyboard region
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract SIFT features
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, mask)
        
        # Cluster keypoints using DBSCAN
        if len(keypoints) > 0:
            points = np.float32([kp.pt for kp in keypoints])
            clustering = DBSCAN(eps=30, min_samples=5).fit(points)
            labels = clustering.labels_
            
            # Filter out noise points
            valid_points = points[labels != -1]
            return valid_points
        return None
        
    def refine_segmentation(self, image, mask, features):
        """
        Refine segmentation using GrabCut and feature points
        """
        if features is None:
            return mask
            
        # Create GrabCut mask
        grabcut_mask = np.zeros(image.shape[:2], np.uint8)
        grabcut_mask[mask > 0] = cv2.GC_PR_FGD
        
        # Create feature mask
        feature_mask = np.zeros_like(grabcut_mask)
        for pt in features.astype(int):
            cv2.circle(feature_mask, tuple(pt), 5, 255, -1)
            
        # Update GrabCut mask based on features
        grabcut_mask[feature_mask > 0] = cv2.GC_FGD
        
        # Run GrabCut
        bgd_model = np.zeros((1,65), np.float64)
        fgd_model = np.zeros((1,65), np.float64)
        
        cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 
                   5, cv2.GC_INIT_WITH_MASK)
                   
        # Create final mask
        final_mask = np.where(
            (grabcut_mask==cv2.GC_FGD)|(grabcut_mask==cv2.GC_PR_FGD), 1, 0
        ).astype('uint8')
        
        return final_mask
        
    def get_optimal_boundary(self, mask, features):
        """
        Get optimal boundary using convex hull and active contours
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
        if not contours:
            return None
            
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute convex hull
        hull = ConvexHull(largest_contour.reshape(-1, 2))
        hull_points = largest_contour.reshape(-1, 2)[hull.vertices]
        
        # Apply active contours
        snake = cv2.snake(mask, hull_points.astype(np.float32), 
                         alpha=0.1, beta=0.1, gamma=0.1)
        
        return snake
        
    def detect(self, image):
        """
        Main detection pipeline
        """
        # Enhance image
        enhanced = self.enhance_image(image)
        
        # Run YOLOv8 segmentation
        results = self.model(enhanced, stream=True)
        
        detections = []
        for result in results:
            # Get masks for keyboard class
            keyboard_masks = result.masks.data[result.boxes.cls == 66]
            conf_scores = result.boxes.conf[result.boxes.cls == 66]
            
            # Process each detection
            for mask, conf in zip(keyboard_masks, conf_scores):
                if conf < self.conf_threshold:
                    continue
                    
                # Convert mask to numpy
                mask = mask.cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                # Extract features
                features = self.extract_features(enhanced, binary_mask)
                
                # Refine segmentation
                refined_mask = self.refine_segmentation(
                    enhanced, binary_mask, features)
                    
                # Get optimal boundary
                boundary = self.get_optimal_boundary(refined_mask, features)
                
                if boundary is not None:
                    # Get bounding box from boundary
                    x_min, y_min = np.min(boundary, axis=0)
                    x_max, y_max = np.max(boundary, axis=0)
                    
                    detections.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'boundary': boundary,
                        'confidence': float(conf),
                        'mask': refined_mask
                    })
        
        return detections
        
    def visualize(self, image, detections):
        """
        Visualize detections with boundaries and masks
        """
        vis_img = image.copy()
        
        for det in detections:
            # Draw boundary
            boundary = det['boundary'].astype(np.int32)
            cv2.polylines(vis_img, [boundary], True, (0, 255, 0), 2)
            
            # Draw mask overlay
            mask = det['mask']
            overlay = vis_img.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([0, 0, 255]) * 0.3
            vis_img = cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0)
            
        return vis_img

# Example usage
if __name__ == "__main__":
    detector = AdvancedKeyboardDetector()
    image = cv2.imread('26.jpg')
    detections = detector.detect(image)
    result = detector.visualize(image, detections)
    cv2.imshow('Advanced Keyboard Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()