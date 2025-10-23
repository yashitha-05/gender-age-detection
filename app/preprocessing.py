import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(image):
    """Detect faces and return bounding boxes."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    return faces

def crop_face(image, box, margin=20):
    """Crop a single face from the image."""
    x, y, w, h = box
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)
    return image[y1:y2, x1:x2]
