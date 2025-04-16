import face_recognition
import numpy as np
import cv2

def extract_face_encoding(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')  # or cnn if GPU
    encodings = face_recognition.face_encodings(rgb, boxes)

    if len(encodings) > 0:
        return encodings[0]  # Trả về vector đầu tiên
    return None
