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
 
# Thu thập ảnh từ web cam

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

dataset_dir = "dataset"
user_folder = os.path.join(dataset_dir, f"{user_name}_{user_id}")
os.makedirs(user_folder, exist_ok=True)

max_images = 100
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        filename = os.path.join(user_folder, f"User.{user_id}.{count}.jpg")
        cv2.imwrite(filename, face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{count}/{max_images}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Add Face", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nĐã lưu {count} ảnh vào {user_folder}")

# Resize ảnh đã thu
source_dir = 'D:/Project VXL/dataset'
target_dir = 'D:/Project VXL/dataset_save'
os.makedirs(target_dir, exist_ok=True)

for user_folder in os.listdir(source_dir):
    user_path = os.path.join(source_dir, user_folder)
    if not os.path.isdir(user_path):
        continue

    save_path = os.path.join(target_dir, user_folder)
    os.makedirs(save_path, exist_ok=True)

    for file in os.listdir(user_path):
        img_path = os.path.join(user_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Lỗi đọc ảnh: {img_path}")
            continue

        resized = cv2.resize(img, (200, 200))
        save_file = os.path.join(save_path, f"resized_{file}")
        cv2.imwrite(save_file, resized)

print("\nĐã resize toàn bộ ảnh và lưu vào thư mục dataset_save.")