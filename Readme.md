# Keyboard Point Detection Project

## Overview

This project is designed to detect and analyze the positions of keys on a keyboard using various image processing and machine learning techniques.

## Scripts

### `detector.py`
Class to take a keyboard find its esitmate bound then tighten them (not done.. probably..using mask-rcnn pixel ml model wise detection) then finding transform to using pnp and applying to all keypoints nown relative to width and breadth of keyboard, storing them then further pass on to the commander
- Not good .. prone to errors .. trainging yolov model on label on keys (bounding box is smaller than keys itself .. size of text)


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

## Usage

Instructions on how to run each script and the overall workflow will be provided here.

## License

This project is licensed under the MIT License.
