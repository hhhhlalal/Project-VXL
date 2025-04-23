import cv2
import os
import sys

# Nhập thông tin từ người dùng
try:
    user_id = input("Nhập ID người dùng: ")
    user_name = input("Nhập tên người dùng: ")

    if not user_id or not user_name:
        raise ValueError("ID hoặc tên người dùng không hợp lệ.")
except Exception as e:
    print(f"Lỗi: {e}")
    sys.exit(1)

# Cấu hình thư mục lưu dữ liệu
dataset_dir = "D:/Project VXL/dataset"
user_folder = os.path.join(dataset_dir, f"{user_name}_{user_id}")
os.makedirs(user_folder, exist_ok=True)
print(f"📁 Ảnh sẽ được lưu tại: {user_folder}")

# Khởi tạo webcam và bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không thể mở webcam.")
    sys.exit(1)

max_images = 200
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc từ webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    print(f"🔍 Phát hiện {len(faces)} khuôn mặt trong khung hình.")

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        filename = os.path.join(user_folder, f"User.{user_id}.{count}.jpg")
        cv2.imwrite(filename, face_img)
        print(f"✅ Đã lưu ảnh: {filename}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{count}/{max_images}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Thêm khuôn mặt", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n📸 Đã lưu tổng cộng {count} ảnh vào thư mục: {user_folder}")
