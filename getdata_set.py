import cv2
import os


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

# Tạo thư mục để lưu khuôn mặt nếu chưa tồn tại
output_dir = "nv 1"   
os.makedirs(output_dir, exist_ok=True)

# Số lượng khuôn mặt đã phát hiện
face_count = 0
# Số lượng ảnh tối đa cho mỗi khuôn mặt
max_images_per_face = 201

# Vòng lặp để đọc và xử lý hình ảnh từ webcam
while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    # Nếu đọc không thành công, thoát vòng lặp
    if not ret:
        break

    # Phát hiện khuôn mặt trong khung hình
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Lặp qua các khuôn mặt đã phát hiện
    for x, y, w, h in faces:
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Kiểm tra số lượng ảnh đã lưu cho khuôn mặt hiện tại
        if face_count < max_images_per_face:
        # Cắt và lưu khuôn mặt vào ổ đĩa
            face = frame[y : y + h, x : x + w]
            face_filename = os.path.join(output_dir, f"face_{face_count}.jpg")
            cv2.imwrite(face_filename, face)

        # Tăng số lượng khuôn mặt đã phát hiện
        face_count += 1

    # Hiển thị khung hình từ webcam
    cv2.imshow("Webcam", frame)

    # Nhấn phím 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng webcam và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
