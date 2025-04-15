import cv2
import os

# Load cascade nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# Nhập thông tin người dùng
user_id = input("Nhập ID người dùng: ")
user_name = input("Nhập tên người dùng: ")

# Tạo thư mục lưu dataset nếu chưa có
dataset_dir = "dataset"
user_folder = os.path.join(dataset_dir, f"{user_name}_{user_id}")
os.makedirs(user_folder, exist_ok=True)

# Thiết lập số lượng ảnh cần lưu
max_images = 100
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc từ webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))  # chuẩn hóa kích thước

        # Lưu ảnh với định dạng User.ID.Num.jpg
        filename = os.path.join(user_folder, f"User.{user_id}.{count}.jpg")
        cv2.imwrite(filename, face_img)

        # Vẽ khung và hiển thị số ảnh đã chụp
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Image {count}/{max_images}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Add New Face", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if count >= max_images:
        print(f"Đã thu thập đủ {max_images} ảnh.")
        break

cap.release()
cv2.destroyAllWindows()
