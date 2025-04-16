from vector_database import load_all_encodings
import face_recognition
import numpy as np

def match_face(input_encoding, threshold=0.45):
    known_faces = load_all_encodings()
    if not known_faces:
        return "Unknown"

    names = [name for name, _ in known_faces]
    encodings = [enc for _, enc in known_faces]

    distances = face_recognition.face_distance(encodings, input_encoding)
    min_dist = np.min(distances)
    best_match_index = np.argmin(distances)

    if min_dist < threshold:
        return names[best_match_index]
    return "Unknown"
