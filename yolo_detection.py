import cv2
import numpy as np
from ultralytics import YOLO

def preprocess_region(image):
    resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    background_subtracted = cv2.subtract(gray, blur)
         
    normalized = cv2.normalize(background_subtracted, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(normalized)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    illumination = cv2.morphologyEx(clahe_result, cv2.MORPH_CLOSE, kernel)
    illumination_corrected = cv2.divide(clahe_result, illumination, scale=255)

    # Apply thresholding to remove unwanted features and noise
    _, thresholded = cv2.threshold(illumination_corrected, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded = cv2.medianBlur(thresholded, 5)

    _, binarized = cv2.threshold(thresholded, 0, 255, cv2.THRESH_BINARY)
    binarized = cv2.GaussianBlur(binarized, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binarized = cv2.dilate(binarized, kernel, iterations=1)
    
    return binarized

def detect_objects_in_image(image_path, model_path='yolov8x.pt', scale_factor=1, target_class=66):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    original_size = image.shape[:2]
    scaled_size = (int(original_size[1] * scale_factor), int(original_size[0] * scale_factor))
    scaled_image = cv2.resize(image, scaled_size, interpolation=cv2.INTER_AREA)
    
    results = model(scaled_image, verbose=False)[0]
    
    for det in results.boxes.data.tolist():
        cls = int(det[5])
        if cls == target_class:
            x1, y1, x2, y2, conf, _ = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(scaled_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Class {cls}: {conf:.2f}"
            cv2.putText(scaled_image, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # region = scaled_image[y1:y2, x1:x2]
            # processed = preprocess_region(region)
            # cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
            # cv2.imshow('Processed', processed)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # if contours:
            #     largest_contour = max(contours, key=cv2.contourArea)
            #     approx = cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, True), True)
            #     epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            #     approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            #     rect = cv2.minAreaRect(largest_contour)
            #     box = cv2.boxPoints(rect)
            #     box = np.int0(box)  # Convert to integer
            #     cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
            #     cv2.imshow("Tightest Bound", image)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()


            #     print(f"Approx: {len(approx)}")
            #     tight_bounds = approx.reshape(-1, 2)
            #     # tight_bounds = tight_bounds * (x2 - x1, y2 - y1) / scaled_image.shape[:2]
            #     tight_bounds += np.array([x1, y1])
            #     cv2.polylines(scaled_image, [tight_bounds], True, (255, 0, 0), 2)
            
            # # epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            # # approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    
    return scaled_image

def main():
    image_path = '26.jpg'
    try:
        annotated_image = detect_objects_in_image(image_path, scale_factor=0.5)
        # cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        
        cv2.imshow('Object Detection', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
