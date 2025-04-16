import cv2
from feature_extractor import extract_face_encoding
from face_matcher import match_face

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    encoding = extract_face_encoding(frame)
    if encoding is not None:
        name = match_face(encoding)
        cv2.putText(frame, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
