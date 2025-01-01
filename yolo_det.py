import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


def guided_filter(I, p, radius, eps):
    """
    Perform guided filtering on image I using guide image p.
    Args:
    - I: Guide image (grayscale or single channel).
    - p: Input image to be filtered.
    - radius: Local window radius.
    - eps: Regularization parameter.
    Returns:
    - The filtered image.
    """
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
    corr_I = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
    
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    
    q = mean_a * I + mean_b
    return q


def detect_advanced_keyboard(image_path, model_path='yolov8x-seg.pt'):
    # Load YOLO model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path)
    conf_threshold = 0.875

    # Load image
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (720, 384))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Image enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Run YOLOv8 segmentation
    results = model(enhanced_image, stream=True)
    detections = []

    for result in results:
        keyboard_masks = result.masks.data[result.boxes.cls == 66]
        conf_scores = result.boxes.conf[result.boxes.cls == 66]

        for mask, conf in zip(keyboard_masks, conf_scores):
            if conf < conf_threshold:
                continue

            mask = mask.cpu().numpy()
            binary_mask = (mask > 0.5).astype(np.uint8)

            # Feature extraction
            gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, binary_mask)
            features = None
            if keypoints:
                points = np.float32([kp.pt for kp in keypoints])
                clustering = DBSCAN(eps=30, min_samples=5).fit(points)
                labels = clustering.labels_
                features = points[labels != -1] if np.any(labels != -1) else None

            # Refine segmentation
            if features is not None:
                # Resize binary_mask to match image dimensions
                binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                grabcut_mask = np.zeros(image.shape[:2], np.uint8)
                grabcut_mask[binary_mask > 0] = cv2.GC_PR_FGD
                for pt in features.astype(int):
                    cv2.circle(grabcut_mask, tuple(pt), 5, cv2.GC_FGD, -1)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                refined_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
            else:
                refined_mask = binary_mask


            # Get optimal boundary
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = ConvexHull(largest_contour.reshape(-1, 2))
                hull_points = largest_contour.reshape(-1, 2)[hull.vertices]
                boundary = hull_points

                x_min, y_min = np.min(boundary, axis=0)
                x_max, y_max = np.max(boundary, axis=0)
                detections.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'boundary': boundary,
                    'confidence': float(conf),
                    'mask': refined_mask
                })

    # Visualization
    vis_img = image.copy()
    for det in detections:
        boundary = det['boundary'].astype(np.int32)
        cv2.polylines(vis_img, [boundary], True, (0, 255, 0), 2)
        mask = det['mask']
        overlay = vis_img.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([0, 0, 255]) * 0.3
        vis_img = cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0)

    return vis_img

# Example usage
if __name__ == "__main__":
    result_image = detect_advanced_keyboard('26.jpg')
    
    cv2.imshow('Advanced Keyboard Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
