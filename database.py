import numpy as np
import sqlite3
import pickle
from typing import List, Tuple, Optional, Dict
import os

class FaceFeatureDatabase:
    def __init__(self, db_path: str = "face_features.db"):
        self.db_path = db_path
        self._initialize_db()
    def _initialize_db(self):
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
    
    def get_all_features(self) -> List[Tuple[int, str, np.ndarray]]:
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM face_features WHERE id = ?', (face_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    def get_feature_vectors_dict(self) -> Dict[str, List[np.ndarray]]:
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM face_features')
        conn.commit()
        conn.close()
