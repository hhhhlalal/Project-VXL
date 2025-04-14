import os
import sqlite3
import pickle
import numpy as np
import json
from typing import List, Tuple, Optional, Dict
import time

class FaceDatabase:
    """
    Database for storing and retrieving face feature vectors
    """
    
    def __init__(self, db_path="face_features.db"):
        """
        Initialize the database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for storing face feature vectors
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            feature_vector BLOB NOT NULL,
            image_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create index on name for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_name ON faces(name)')
        
        conn.commit()
        conn.close()
    
    def add_face(self, name: str, feature_vector: np.ndarray, image_path: Optional[str] = None) -> int:
        """
        Add a face to the database
        
        Args:
            name: Person's name
            feature_vector: Feature vector
            image_path: Path to source image (optional)
            
        Returns:
            ID of the new record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize the numpy array
        serialized_vector = pickle.dumps(feature_vector)
        
        cursor.execute(
            'INSERT INTO faces (name, feature_vector, image_path) VALUES (?, ?, ?)',
            (name, serialized_vector, image_path)
        )
        
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return face_id
    
    def get_all_faces(self) -> List[Tuple[int, str, np.ndarray, Optional[str]]]:
        """
        Get all faces from the database
        
        Returns:
            List of (id, name, feature_vector, image_path) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, feature_vector, image_path FROM faces')
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            face_id, name, serialized_vector, image_path = row
            feature_vector = pickle.loads(serialized_vector)
            result.append((face_id, name, feature_vector, image_path))
        
        conn.close()
        return result
    
    def get_faces_by_name(self, name: str) -> List[Tuple[int, np.ndarray, Optional[str]]]:
        """
        Get faces for a specific person
        
        Args:
            name: Person's name
            
        Returns:
            List of (id, feature_vector, image_path) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, feature_vector, image_path FROM faces WHERE name = ?',
            (name,)
        )
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            face_id, serialized_vector, image_path = row
            feature_vector = pickle.loads(serialized_vector)
            result.append((face_id, feature_vector, image_path))
        
        conn.close()
        return result
    
    def get_unique_names(self) -> List[str]:
        """
        Get list of all unique names in the database
        
        Returns:
            List of names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT name FROM faces')
        names = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return names
    
    def get_face_by_id(self, face_id: int) -> Optional[Tuple[str, np.ndarray, Optional[str]]]:
        """
        Get face by ID
        
        Args:
            face_id: Face ID
            
        Returns:
            Tuple (name, feature_vector, image_path) or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT name, feature_vector, image_path FROM faces WHERE id = ?',
            (face_id,)
        )
        row = cursor.fetchone()
        
        if row:
            name, serialized_vector, image_path = row
            feature_vector = pickle.loads(serialized_vector)
            conn.close()
            return (name, feature_vector, image_path)
        
        conn.close()
        return None
    
    def update_face(self, face_id: int, name: str = None, feature_vector: np.ndarray = None) -> bool:
        """
        Update a face in the database
        
        Args:
            face_id: Face ID
            name: New name (optional)
            feature_vector: New feature vector (optional)
            
        Returns:
            True if updated, False if not found
        """
        if name is None and feature_vector is None:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if name is not None and feature_vector is not None:
            serialized_vector = pickle.dumps(feature_vector)
            cursor.execute(
                'UPDATE faces SET name = ?, feature_vector = ? WHERE id = ?',
                (name, serialized_vector, face_id)
            )
        elif name is not None:
            cursor.execute(
                'UPDATE faces SET name = ? WHERE id = ?',
                (name, face_id)
            )
        else:  # feature_vector is not None
            serialized_vector = pickle.dumps(feature_vector)
            cursor.execute(
                'UPDATE faces SET feature_vector = ? WHERE id = ?',
                (serialized_vector, face_id)
            )
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def delete_face(self, face_id: int) -> bool:
        """
        Delete a face from the database
        
        Args:
            face_id: Face ID
            
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def get_face_count(self) -> int:
        """
        Get total number of faces in the database
        
        Returns:
            Number of faces
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM faces')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def get_person_count(self) -> int:
        """
        Get number of unique persons in the database
        
        Returns:
            Number of unique persons
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(DISTINCT name) FROM faces')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def export_to_json(self, output_path: str) -> bool:
        """
        Export database to JSON file
        
        Args:
            output_path: Path to output JSON file
            
        Returns:
            True if successful
        """
        data = {
            "persons": {},
            "metadata": {
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_faces": self.get_face_count(),
                "total_persons": self.get_person_count()
            }
        }
        
        # Get all unique names
        names = self.get_unique_names()
        
        # For each person, get their faces
        for name in names:
            faces = self.get_faces_by_name(name)
            
            data["persons"][name] = {
                "face_count": len(faces),
                "faces": [
                    {
                        "id": face_id,
                        "image_path": image_path if image_path else "Unknown",
                        # We don't export the actual feature vectors as they're large
                        # and not human-readable
                    }
                    for face_id, _, image_path in faces
                ]
            }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True