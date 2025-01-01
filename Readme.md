# Keyboard Point Detection Project

## Overview

This project is designed to detect and analyze the positions of keys on a keyboard using various image processing and machine learning techniques.

## Scripts

### `detector.py`
Class to take a keyboard find its esitmate bound then tighten them (not done.. probably..using mask-rcnn pixel ml model wise detection) then finding transform to using pnp and applying to all keypoints nown relative to width and breadth of keyboard, storing them then further pass on to the commander
- Not good .. prone to errors .. trainging yolov model on label on keys (bounding box is smaller than keys itself .. size of text .. mid point of bounding box is likely the point to target)

### `commander.py`
Its pseudo implementation of idea of how commander works

### `yolov_trainner.py`
To train a model on dataset `augmented_dataset_yolo` - 7000 images .. synthetically made using 11 images that are annotated for 75 different classes

### `annotations.py` & `annotation_2.py` & `GAN.py`
To augment dataset for yolo training

### `yolo_det.py` & `yolo_detection_2.py` & `yolo_detection.py`
fuck this shit .. pretrained model wont cut it

### `final.py`

Calculates the 3D positions of all keys on the keyboard. It integrates data from previous processing steps to accurately determine the spatial coordinates of each key, enabling precise mapping and analysis.

### `point_marker.py`
Identifies and marks the key points on the keyboard.

### `preprocessing_for_labeling.py`
Preprocesses images to prepare them for labeling, ensuring accurate annotations.

### `image_preprocessing.py`
Processes images to produce well-binarized images using the Otsu method for enhanced clarity.

### `template_matching.py`
Attempts to locate keys and the keyboard using template matching on the well-defined keyboard structure. This approach did not yield the desired results.

### `yolo_detection.py`
Utilizes YOLO for detecting the four corners of the keyboard (class 66). However, it does not produce perfect bounding boxes as required.



### `annotation.py`

## Note

OCR will not work .. maybe you could make it but its intended use is for world level detection after that it segments the word into characters

## License

This project is licensed under the MIT License.
