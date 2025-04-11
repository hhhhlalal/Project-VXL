import os
import numpy as np
import dlib
import cv2

class FeatureExtractor:
    def __init__(self, model_path="../models/"):
        # Load dlib face recognition model
        recognition_model_path = os.path.join(model_path, "facenet.pb")
        self.face_recognition_model = dlib.face_recognition_model_v1(recognition_model_path)
        
        # Face detector will be used if needed
        self.face_detector = None
        self.shape_predictor = None
    
    def _ensure_detector_loaded(self, model_path="../models/"):
        if self.face_detector is None:
            self.face_detector = dlib.get_frontal_face_detector()
            
            predictor_path = os.path.join(model_path, "shape_predictor.dat")
            self.shape_predictor = dlib.shape_predictor(predictor_path)
    
    def compute_feature_vector(self, image, face_rect=None):
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # If no face rectangle provided, assume the entire image is a face
        # or detect the face
        if face_rect is None:
            # Check if image is already a cropped face
            h, w = rgb_image.shape[:2]
            if max(h, w) > 300:  # Likely not a cropped face
                self._ensure_detector_loaded()
                faces = self.face_detector(rgb_image, 1)
                if len(faces) == 0:
                    raise ValueError("No face detected in image")
                face_rect = faces[0]
            else:
                # Create a rectangle covering the whole image
                face_rect = dlib.rectangle(0, 0, w, h)
        elif isinstance(face_rect, tuple):
            # Convert (x, y, w, h) to dlib rectangle
            x, y, w, h = face_rect
            face_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get face landmarks
        self._ensure_detector_loaded()
        shape = self.shape_predictor(rgb_image, face_rect)
        
        # Compute feature vector
        face_descriptor = self.face_recognition_model.compute_face_descriptor(rgb_image, shape)
        
        # Convert to numpy array
        return np.array(face_descriptor)
    
    def extract_features_from_image(self, image_path):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        self._ensure_detector_loaded()
        face_dets = self.face_detector(rgb_image, 1)
        
        results = []
        for face_det in face_dets:
            # Convert dlib rectangle to (x, y, w, h)
            x = face_det.left()
            y = face_det.top()
            w = face_det.right() - x
            h = face_det.bottom() - y
            face_rect = (x, y, w, h)
            
            # Get landmarks
            shape = self.shape_predictor(rgb_image, face_det)
            
            # Compute feature vector
            feature_vector = self.face_recognition_model.compute_face_descriptor(rgb_image, shape)
            feature_vector = np.array(feature_vector)
            
            results.append((face_rect, feature_vector))
        
        return results
    
    def compute_similarity(self, feature1, feature2):
        # Calculate Euclidean distance
        distance = np.linalg.norm(feature1 - feature2)
        
        # Convert to similarity score (0-1)
        similarity = max(0, min(1, 1 - distance / 2))
        
        return similarity