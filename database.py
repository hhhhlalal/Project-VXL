
import numpy as np
import sqlite3
import pickle
import os
import dlib
import cv2
from typing import List, Tuple, Optional, Dict

class FeatureExtractor:
    """
    Class for extracting face features using Dlib
    """
    
    def __init__(self):
        """Initialize face detection and recognition models."""
        # Load dlib models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # HOG-based face detector
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Paths to model files
        predictor_path = os.path.join(current_dir, "models", "shape_predictor_68_face_landmarks.dat")
        recognition_model_path = os.path.join(current_dir, "models", "dlib_face_recognition_resnet_model_v1.dat")
        
        # Facial landmark predictor
        self.shape_predictor = dlib.shape_predictor(predictor_path)
        
        # Face recognition model (pre-trained ResNet)
        self.face_recognition_model = dlib.face_recognition_model_v1(recognition_model_path)
    
    def detect_faces(self, image):
        """
        Detect all faces in an image
        
        Args:
            image: Input image (OpenCV format)
            
        Returns:
            List of face rectangles
        """
        # Convert from BGR to RGB (dlib needs RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Detect faces
        faces = self.face_detector(rgb_image, 1)
        
        # Convert dlib results to list of rectangles (x, y, w, h)
        face_rects = []
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            face_rects.append((x, y, w, h))
            
        return face_rects
    
    def get_landmarks(self, image, face):
        """
        Retrieve 68 facial landmarks for a detected face
        
        Args:
            image: Input image
            face: Face rectangle from detector
            
        Returns:
            Facial landmarks
        """
        # Convert to dlib rectangle format
        x, y, w, h = face
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Convert from BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Get face landmarks
        shape = self.shape_predictor(rgb_image, dlib_rect)
        
        return shape
    
    def compute_feature_vector(self, image, shape):
        """
        Compute 128-D feature vector for a face
        
        Args:
            image: Input image
            shape: Shape object from dlib
            
        Returns:
            128-dimensional feature vector
        """
        # Convert from BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Compute 128-dimensional feature vector
        face_descriptor = self.face_recognition_model.compute_face_descriptor(rgb_image, shape)
        
        # Convert to numpy array
        return np.array(face_descriptor)
    
    def extract_features_from_image(self, image):
        """
        Detect all faces and extract features from an image
        
        Args:
            image: Input image
            
        Returns:
            List of tuples (face, feature_vector)
        """
        results = []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        for face in faces:
            # Get landmarks
            shape = self.get_landmarks(image, face)
            
            # Compute feature vector
            descriptor = self.compute_feature_vector(image, shape)
            
            results.append((face, descriptor))
        
        return results


class FaceFeaturesDB:
    """
    Database for storing and retrieving face feature vectors along with associated names.
    This implementation uses SQLite for persistent storage with serialized numpy arrays.
    """
    
    def __init__(self, db_path: str = "face_features.db"):
        """Initialize the face feature database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
        
        # Initialize dlib feature extractor
        try:
            self.extractor = FeatureExtractor()
            self.has_dlib = True
        except Exception as e:
            print(f"Could not load dlib models: {e}")
            self.has_dlib = False
        
    def _initialize_db(self):
        """Create the database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for storing face feature vectors
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            feature_vector BLOB NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create index on name for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_name ON face_features(name)')
        
        conn.commit()
        conn.close()
    
    def add_face(self, name: str, feature_vector: np.ndarray) -> int:
        """Add a new face feature vector to the database.
        
        Args:
            name: Name of the person
            feature_vector: Numpy array containing facial features
            
        Returns:
            ID of the inserted record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize the numpy array
        serialized_vector = pickle.dumps(feature_vector)
        
        cursor.execute(
            'INSERT INTO face_features (name, feature_vector) VALUES (?, ?)',
            (name, serialized_vector)
        )
        
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return face_id
    
    def add_face_from_image(self, name: str, image) -> List[int]:
        """Add face feature vectors from an image.
        
        Args:
            name: Name of the person
            image: Image containing face (OpenCV format)
            
        Returns:
            List of IDs of the inserted records
        """
        if not self.has_dlib:
            raise ValueError("Dlib is not available, cannot extract features from image")
            
        # Extract face features from image
        face_features = self.extractor.extract_features_from_image(image)
        
        ids = []
        for _, descriptor in face_features:
            # Add each face to the database
            face_id = self.add_face(name, descriptor)
            ids.append(face_id)
            
        return ids
    
    def get_all_features(self) -> List[Tuple[int, str, np.ndarray]]:
        """Retrieve all face features from the database.
        
        Returns:
            List of tuples containing (id, name, feature_vector)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, feature_vector FROM face_features')
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            face_id, name, serialized_vector = row
            feature_vector = pickle.loads(serialized_vector)
            result.append((face_id, name, feature_vector))
        
        conn.close()
        return result
    
    def get_features_by_name(self, name: str) -> List[Tuple[int, np.ndarray]]:
        """Retrieve face features for a specific person.
        
        Args:
            name: Name of the person to search for
            
        Returns:
            List of tuples containing (id, feature_vector) for the given name
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, feature_vector FROM face_features WHERE name = ?', 
            (name,)
        )
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            face_id, serialized_vector = row
            feature_vector = pickle.loads(serialized_vector)
            result.append((face_id, feature_vector))
        
        conn.close()
        return result
    
    def get_feature_by_id(self, face_id: int) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Retrieve a face feature by its ID.
        
        Args:
            face_id: ID of the face feature record
            
        Returns:
            Tuple of (name, feature_vector) or (None, None) if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT name, feature_vector FROM face_features WHERE id = ?', 
            (face_id,)
        )
        row = cursor.fetchone()
        
        if row:
            name, serialized_vector = row
            feature_vector = pickle.loads(serialized_vector)
            conn.close()
            return (name, feature_vector)
        
        conn.close()
        return (None, None)
    
    def update_face(self, face_id: int, name: str, feature_vector: np.ndarray) -> bool:
        """Update an existing face feature vector.
        
        Args:
            face_id: ID of the face feature record
            name: New name for the person
            feature_vector: New feature vector
            
        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        serialized_vector = pickle.dumps(feature_vector)
        
        cursor.execute(
            'UPDATE face_features SET name = ?, feature_vector = ? WHERE id = ?',
            (name, serialized_vector, face_id)
        )
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def delete_face(self, face_id: int) -> bool:
        """Delete a face feature from the database.
        
        Args:
            face_id: ID of the face feature record
            
        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM face_features WHERE id = ?', (face_id,))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def get_feature_vectors_dict(self) -> Dict[str, List[np.ndarray]]:
        """Get all feature vectors organized by name.
        
        Returns:
            Dictionary with names as keys and lists of feature vectors as values
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT name, feature_vector FROM face_features')
        rows = cursor.fetchall()
        
        result = {}
        for row in rows:
            name, serialized_vector = row
            feature_vector = pickle.loads(serialized_vector)
            
            if name not in result:
                result[name] = []
            
            result[name].append(feature_vector)
        
        conn.close()
        return result
    
    def clear_database(self) -> None:
        """Clear all records from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM face_features')
        
        conn.commit()
        conn.close()
    
    def recognize_face(self, image, threshold: float = 0.6) -> List[Tuple[str, float, tuple]]:
        """Recognize faces in an image by comparing with the database.
        
        Args:
            image: Input image
            threshold: Distance threshold to determine if faces match
            
        Returns:
            List of tuples (name, confidence, face_rectangle)
        """
        if not self.has_dlib:
            raise ValueError("Dlib is not available, cannot recognize faces")
        
        # Get all known vectors and names
        vector_dict = self.get_feature_vectors_dict()
        
        # Detect and extract features from input image
        face_features = self.extractor.extract_features_from_image(image)
        
        results = []
        
        for face_rect, unknown_descriptor in face_features:
            best_match = "Unknown"
            best_distance = float('inf')
            
            # Compare with each known face
            for name, descriptors in vector_dict.items():
                for known_descriptor in descriptors:
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(unknown_descriptor - known_descriptor)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name
            
            # Convert distance to confidence score (0-1)
            confidence = max(0, min(1, 1 - best_distance))
            
            # Only return results if confidence exceeds threshold
            if confidence >= threshold:
                results.append((best_match, confidence, face_rect))
            else:
                results.append(("Unknown", confidence, face_rect))
                
        return results
