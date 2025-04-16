import os
import cv2

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

print("Đã resize toàn bộ ảnh vào thư mục dataset_save.")
