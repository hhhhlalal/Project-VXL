import os
import cv2
from feature_extractor import extract_face_encoding
from vector_database import create_db, insert_face

create_db()

for person in os.listdir("dataset_save"):
    person_path = os.path.join("dataset_save", person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            _, encoding = extract_face_encoding(img)
            if encoding is not None:
                insert_face(person, encoding)
                print(f"Đã thêm vector của {person} từ {img_name}")