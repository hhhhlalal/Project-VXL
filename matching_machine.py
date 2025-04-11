import os
import numpy as np
import cv2
from typing import List, Dict

class FaceMatcher:
    """
    Class for matching faces against a database of known faces
    """
    
    def __init__(self, feature_extractor, threshold=0.6):
        """
        Initialize face matcher
        
        Args:
            feature_extractor: FeatureExtractor instance
            threshold: Similarity threshold for face matching (0-1)
        """
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.known_faces = {}  # name -> list of feature vectors
    
    def add_known_face(self, name, feature_vector):
        """
        Add a known face to the matcher
        
        Args:
            name: Person's name
            feature_vector: Face feature vector
        """
        if name not in self.known_faces:
            self.known_faces[name] = []
        
        self.known_faces[name].append(feature_vector)
    
    def add_known_face_from_image(self, name, image_path):
        """
        Add a known face from an image
        
        Args:
            name: Person's name
            image_path: Path to face image
            
        Returns:
            Number of faces added
        """
        # Extract features from image
        results = self.feature_extractor.extract_features_from_image(image_path)
        
        # Add each face
        for _, feature_vector in results:
            self.add_known_face(name, feature_vector)
        
        return len(results)
    
    def load_known_faces_directory(self, directory):
        """
        Load known faces from a directory structure
        Directory structure should be: directory/person_name/image_files
        
        Args:
            directory: Root directory for known faces
            
        Returns:
            Number of faces loaded
        """
        total_faces = 0
        
        # Iterate through person directories
        for person_name in os.listdir(directory):
            person_dir = os.path.join(directory, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            # Iterate through images in person directory
            for image_file in os.listdir(person_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, image_file)
                    
                    try:
                        # Add face from image
                        num_faces = self.add_known_face_from_image(person_name, image_path)
                        total_faces += num_faces
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
        
        return total_faces
    
    def match_face(self, feature_vector):
        """
        Match a face against known faces
        
        Args:
            feature_vector: Feature vector to match
            
        Returns:
            Tuple (name, confidence) of best match, or ("Unknown", 0) if no match
        """
        best_match = "Unknown"
        best_confidence = 0
        
        # Compare with each known face
        for name, feature_vectors in self.known_faces.items():
            for known_vector in feature_vectors:
                # Compute similarity
                similarity = self.feature_extractor.compute_similarity(feature_vector, known_vector)
                
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_match = name
        
        # Check if confidence exceeds threshold
        if best_confidence < self.threshold:
            return "Unknown", best_confidence
        
        return best_match, best_confidence
    
    def recognize_faces_in_image(self, image_path):
        """
        Recognize all faces in an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of (face_rect, name, confidence) tuples
        """
        # Extract features from image
        face_features = self.feature_extractor.extract_features_from_image(image_path)
        
        results = []
        for face_rect, feature_vector in face_features:
            # Match face
            name, confidence = self.match_face(feature_vector)
            
            results.append((face_rect, name, confidence))
        
        return results