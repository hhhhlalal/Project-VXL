import sqlite3
import numpy as np
import os

def create_db(db_path="face_db.sqlite"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    encoding BLOB NOT NULL
                 )''')
    conn.commit()
    conn.close()

def insert_face(name, encoding, db_path="face_db.sqlite"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    encoding_blob = encoding.tobytes()
    c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_blob))
    conn.commit()
    conn.close()

def load_all_encodings(db_path="face_db.sqlite"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM faces")
    results = []
    for name, encoding_blob in c.fetchall():
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        results.append((name, encoding))
    conn.close()
    return results
