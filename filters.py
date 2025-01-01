import cv2
print(cv2.__version__)  # Ensure it matches the latest version.
print(hasattr(cv2, 'ximgproc'))  # Should return True if ximgproc is available.
